/* Implementation of the C region API; see scope_profiler.h.
 *
 * The trace format extends the one the Fortran API writes (documented in
 * scope_profiler/native_trace.py) with an optional source location per
 * region, gated on the format version so version-1 (Fortran) files keep
 * reading exactly as before:
 *
 *     char[8]   "SCOPEPRF"
 *     int32     format version (1 or 2)
 *     int32     rank
 *     int64     number of regions
 *     per region:
 *         int32     length of the name in bytes
 *         char[]    name
 *         -- version 2 only --
 *         int32     length of the source file path in bytes (0 if unknown)
 *         char[]    source file path
 *         int32     source line (-1 if unknown)
 *         -- end version 2 only --
 *         int64     number of calls
 *         int64[]   start timestamps, nanoseconds
 *         int64[]   end timestamps, nanoseconds
 */
/* Feature macros, per platform and for opposite reasons.
 *
 * glibc needs _POSIX_C_SOURCE >= 200809 to declare clock_gettime and strdup
 * under a strict -std=c99. macOS needs the *opposite*: defining
 * _POSIX_C_SOURCE there hides CLOCK_UPTIME_RAW, which is a Darwin extension --
 * and that is the only clock on macOS that both matches CPython's
 * perf_counter_ns() epoch and has nanosecond resolution (its CLOCK_MONOTONIC
 * is microsecond-granular and starts from a different origin). Getting this
 * wrong silently costs a factor of 1000 in precision and puts C regions on a
 * different timeline than Python ones. */
#if defined(__APPLE__)
#  define _DARWIN_C_SOURCE
#else
#  define _POSIX_C_SOURCE 200809L
#endif

#include "scope_profiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char SP_MAGIC[8] = {'S', 'C', 'O', 'P', 'E', 'P', 'R', 'F'};

/* Initial slots per region; the buffers double from here as needed. */
#define SP_INITIAL_CAPACITY 1024

/* Deepest recursion of a single region that can be open at once. */
#define SP_MAX_DEPTH 64

/* Deepest total nesting of open calls (any regions) sp_profiler_end_last()
 * can track at once. Overflowing this only degrades sp_profiler_end_last();
 * sp_profiler_begin/sp_profiler_end are unaffected, since they use each
 * region's own open_slots stack instead. */
#define SP_MAX_OPEN 1024

typedef struct {
    char *name;
    char *source_file;  /* NULL if unknown */
    int32_t source_line; /* -1 if unknown */
    int64_t *start_times;
    int64_t *end_times;
    int64_t ptr;      /* slots used */
    int64_t capacity;
    int64_t num_calls;
    /* Slots reserved by calls still open, innermost last, so a recursive
     * re-entry reserves its own slot instead of overwriting the outer one. */
    int64_t open_slots[SP_MAX_DEPTH];
    int depth;
    int64_t total_ns;
    int64_t min_ns;
    int64_t max_ns;
    int64_t stat_calls;  /* completed sp_end()s; distinguishes "no calls yet"
                          * from "one call that happened to take 0ns" for
                          * min_ns/max_ns initialization */
} sp_region_t;

struct sp_profiler {
    sp_region_t *regions;
    int n_regions;
    int regions_capacity;
    int rank_id;
    char *output_prefix;
    char *cached_path;  /* built lazily by trace_path() */
    int active;
    sp_status last_error;
    /* Cross-region stack of open calls, for sp_profiler_end_last(). */
    int open_stack[SP_MAX_OPEN];
    int open_depth;
};

/* Resolved on first use; -1 means "not yet probed", -2 "none works". Shared
 * by every profiler: which clock exists is a property of the machine, not of
 * any one instance.
 *
 * CLOCK_MONOTONIC is what CPython's perf_counter_ns() reads on Linux;
 * CLOCK_UPTIME_RAW is what it reads on macOS. Sharing that clock is what puts
 * C regions and Python regions on one timeline. */
static clockid_t clock_id = (clockid_t)-1;
static int clock_resolved = 0;

/* The profiler behind sp_init()/sp_region()/sp_begin()/sp_end()/sp_finalize(). */
static sp_profiler *g_default = NULL;

static void resolve_clock(void)
{
    static const clockid_t candidates[] = {
#ifdef CLOCK_UPTIME_RAW
        CLOCK_UPTIME_RAW, /* macOS: what perf_counter_ns() uses there */
#endif
        CLOCK_MONOTONIC,
    };
    struct timespec ts;
    size_t i;

    clock_resolved = 1;
    for (i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        if (clock_gettime(candidates[i], &ts) == 0) {
            clock_id = candidates[i];
            return;
        }
    }
    clock_id = (clockid_t)-2;
}

int64_t sp_now_ns(void)
{
    struct timespec ts;

    if (!clock_resolved) {
        resolve_clock();
    }
    if (clock_id == (clockid_t)-2) {
        return -1;
    }
    if (clock_gettime(clock_id, &ts) != 0) {
        return -1;
    }
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

const char *sp_error_string(sp_status status)
{
    switch (status) {
        case SP_OK: return "ok";
        case SP_ERR_INACTIVE: return "profiler is not active";
        case SP_ERR_NO_CLOCK: return "no monotonic clock available";
        case SP_ERR_NO_MEMORY: return "out of memory";
        case SP_ERR_IO: return "trace file could not be written";
        case SP_ERR_UNMATCHED_END: return "no open call to end";
        case SP_ERR_OPEN_SCOPES: return "a region is still open";
    }
    return "unknown status";
}

int sp_profiler_is_active(const sp_profiler *profiler)
{
    return profiler != NULL && profiler->active;
}

sp_status sp_profiler_last_error(const sp_profiler *profiler)
{
    return profiler != NULL ? profiler->last_error : SP_OK;
}

int sp_profiler_num_regions(const sp_profiler *profiler)
{
    return profiler != NULL ? profiler->n_regions : 0;
}

const char *sp_profiler_region_name(const sp_profiler *profiler, int region)
{
    if (profiler == NULL || region < 0 || region >= profiler->n_regions) {
        return NULL;
    }
    return profiler->regions[region].name;
}

/* Release a region's timestamp buffers but keep its name, source location and
 * counters. Used by sp_profiler_finalize(): call counts and stats stay
 * readable afterwards, matching the Fortran and Python APIs. */
static void release_buffers(sp_profiler *profiler)
{
    int i;

    for (i = 0; i < profiler->n_regions; ++i) {
        free(profiler->regions[i].start_times);
        free(profiler->regions[i].end_times);
        profiler->regions[i].start_times = NULL;
        profiler->regions[i].end_times = NULL;
        profiler->regions[i].capacity = 0;
        profiler->regions[i].ptr = 0;
        profiler->regions[i].depth = 0;
    }
    profiler->open_depth = 0;
}

sp_profiler *sp_create(const char *prefix, int rank)
{
    sp_profiler *profiler = (sp_profiler *)calloc(1, sizeof(*profiler));

    if (profiler == NULL) {
        return NULL;
    }

    resolve_clock();
    if (clock_id == (clockid_t)-2) {
        fprintf(stderr,
                "scope_profiler: no monotonic clock available "
                "(clock_gettime rejected every candidate); profiling is disabled\n");
        profiler->last_error = SP_ERR_NO_CLOCK;
        return profiler;
    }

    profiler->output_prefix = strdup(prefix != NULL ? prefix : "scope_profile");
    if (profiler->output_prefix == NULL) {
        fprintf(stderr, "scope_profiler: out of memory in sp_create\n");
        profiler->last_error = SP_ERR_NO_MEMORY;
        return profiler;
    }

    profiler->rank_id = rank;
    profiler->active = 1;
    profiler->last_error = SP_OK;
    return profiler;
}

void sp_destroy(sp_profiler *profiler)
{
    int i;

    if (profiler == NULL) {
        return;
    }
    for (i = 0; i < profiler->n_regions; ++i) {
        free(profiler->regions[i].name);
        free(profiler->regions[i].source_file);
        free(profiler->regions[i].start_times);
        free(profiler->regions[i].end_times);
    }
    free(profiler->regions);
    free(profiler->output_prefix);
    free(profiler->cached_path);
    free(profiler);
}

static int sp_profiler_region_impl(
    sp_profiler *profiler,
    const char *name,
    const char *source_file,
    int source_line)
{
    sp_region_t *bigger;
    sp_region_t *region;
    int i;

    if (profiler == NULL) {
        return SP_INVALID_REGION;
    }
    if (!profiler->active || name == NULL) {
        profiler->last_error = SP_ERR_INACTIVE;
        return SP_INVALID_REGION;
    }

    for (i = 0; i < profiler->n_regions; ++i) {
        if (strcmp(profiler->regions[i].name, name) == 0) {
            /* Backfill: a name first seen via plain sp_region() may be
             * re-registered later with a source location (or vice versa,
             * which is a no-op here) -- take it if this handle does not have
             * one yet. Never overwrites a location it already has. */
            if (source_file != NULL && profiler->regions[i].source_file == NULL) {
                profiler->regions[i].source_file = strdup(source_file);
                profiler->regions[i].source_line =
                    profiler->regions[i].source_file != NULL ? source_line : -1;
            }
            profiler->last_error = SP_OK;
            return i;
        }
    }

    if (profiler->n_regions == profiler->regions_capacity) {
        int capacity = profiler->regions_capacity == 0 ? 16 : profiler->regions_capacity * 2;

        bigger = (sp_region_t *)realloc(
            profiler->regions, (size_t)capacity * sizeof(*profiler->regions));
        if (bigger == NULL) {
            fprintf(stderr, "scope_profiler: out of memory registering '%s'\n", name);
            profiler->last_error = SP_ERR_NO_MEMORY;
            return SP_INVALID_REGION;
        }
        profiler->regions = bigger;
        profiler->regions_capacity = capacity;
    }

    region = &profiler->regions[profiler->n_regions];
    memset(region, 0, sizeof(*region));
    region->name = strdup(name);
    region->source_line = -1;
    region->start_times = (int64_t *)malloc(SP_INITIAL_CAPACITY * sizeof(int64_t));
    region->end_times = (int64_t *)malloc(SP_INITIAL_CAPACITY * sizeof(int64_t));
    if (region->name == NULL || region->start_times == NULL ||
        region->end_times == NULL) {
        free(region->name);
        free(region->start_times);
        free(region->end_times);
        fprintf(stderr, "scope_profiler: out of memory registering '%s'\n", name);
        profiler->last_error = SP_ERR_NO_MEMORY;
        return SP_INVALID_REGION;
    }
    if (source_file != NULL) {
        region->source_file = strdup(source_file);
        region->source_line = region->source_file != NULL ? source_line : -1;
    }
    region->capacity = SP_INITIAL_CAPACITY;

    profiler->last_error = SP_OK;
    return profiler->n_regions++;
}

int sp_profiler_region(sp_profiler *profiler, const char *name)
{
    return sp_profiler_region_impl(profiler, name, NULL, -1);
}

int sp_profiler_region_at(
    sp_profiler *profiler,
    const char *name,
    const char *source_file,
    int source_line)
{
    return sp_profiler_region_impl(profiler, name, source_file, source_line);
}

/* Double a region's timestamp buffers. Returns 0 on success. */
static int grow(sp_region_t *region)
{
    int64_t capacity = region->capacity > 0 ? region->capacity * 2 : 1;
    int64_t *starts;
    int64_t *ends;

    starts = (int64_t *)realloc(region->start_times, (size_t)capacity * sizeof(int64_t));
    if (starts == NULL) {
        return 1;
    }
    region->start_times = starts;

    ends = (int64_t *)realloc(region->end_times, (size_t)capacity * sizeof(int64_t));
    if (ends == NULL) {
        return 1;
    }
    region->end_times = ends;

    region->capacity = capacity;
    return 0;
}

void sp_profiler_begin(sp_profiler *profiler, int region)
{
    sp_region_t *r;

    if (profiler == NULL || !profiler->active || region < 0 || region >= profiler->n_regions) {
        return;
    }
    r = &profiler->regions[region];

    if (r->ptr >= r->capacity && grow(r) != 0) {
        fprintf(stderr, "scope_profiler: out of memory recording '%s'; "
                        "this call is not timed\n",
                r->name);
        profiler->last_error = SP_ERR_NO_MEMORY;
        return;
    }
    r->num_calls += 1;

    if (r->depth >= SP_MAX_DEPTH) {
        fprintf(stderr,
                "scope_profiler: region '%s' nested deeper than %d; "
                "this call is not timed\n",
                r->name, SP_MAX_DEPTH);
        return;
    }

    r->open_slots[r->depth] = r->ptr;
    r->depth += 1;
    r->ptr += 1;
    if (profiler->open_depth < SP_MAX_OPEN) {
        profiler->open_stack[profiler->open_depth] = region;
    }
    profiler->open_depth += 1;
    r->start_times[r->open_slots[r->depth - 1]] = sp_now_ns();
    profiler->last_error = SP_OK;
}

/* End the call currently open in `region` (its innermost, per open_slots),
 * recording it into the running stats. Does not touch the cross-region
 * open_stack; callers pop that themselves. */
static void end_region(sp_profiler *profiler, sp_region_t *region)
{
    int64_t slot;
    int64_t duration;

    region->depth -= 1;
    slot = region->open_slots[region->depth];
    region->end_times[slot] = sp_now_ns();

    duration = region->end_times[slot] - region->start_times[slot];
    if (region->stat_calls == 0) {
        region->min_ns = duration;
        region->max_ns = duration;
    } else {
        if (duration < region->min_ns) {
            region->min_ns = duration;
        }
        if (duration > region->max_ns) {
            region->max_ns = duration;
        }
    }
    region->total_ns += duration;
    region->stat_calls += 1;
    profiler->last_error = SP_OK;
}

void sp_profiler_end(sp_profiler *profiler, int region)
{
    sp_region_t *r;

    if (profiler == NULL || !profiler->active || region < 0 || region >= profiler->n_regions) {
        return;
    }
    r = &profiler->regions[region];

    if (r->depth <= 0) {
        fprintf(stderr, "scope_profiler: sp_end('%s') without a matching sp_begin\n",
                r->name);
        profiler->last_error = SP_ERR_UNMATCHED_END;
        return;
    }

    if (profiler->open_depth > 0) {
        profiler->open_depth -= 1;
    }
    end_region(profiler, r);
}

int sp_profiler_end_last(sp_profiler *profiler)
{
    int region;

    if (profiler == NULL || !profiler->active || profiler->open_depth <= 0) {
        if (profiler != NULL) {
            profiler->last_error = SP_ERR_UNMATCHED_END;
        }
        return SP_ERR_UNMATCHED_END;
    }
    profiler->open_depth -= 1;
    if (profiler->open_depth >= SP_MAX_OPEN) {
        /* This entry's region id was never recorded (stack overflowed while
         * it was pushed); nothing safe to end. */
        profiler->last_error = SP_ERR_UNMATCHED_END;
        return SP_ERR_UNMATCHED_END;
    }
    region = profiler->open_stack[profiler->open_depth];
    if (region < 0 || region >= profiler->n_regions || profiler->regions[region].depth <= 0) {
        profiler->last_error = SP_ERR_UNMATCHED_END;
        return SP_ERR_UNMATCHED_END;
    }
    end_region(profiler, &profiler->regions[region]);
    return SP_OK;
}

sp_scope sp_profiler_scope_begin(sp_profiler *profiler, int region)
{
    sp_scope scope;

    scope.profiler = profiler;
    scope.region = region;
    scope.slot = -1;

    if (profiler == NULL || !profiler->active || region < 0 || region >= profiler->n_regions) {
        return scope;
    }
    sp_profiler_begin(profiler, region);
    if (profiler->regions[region].depth > 0) {
        scope.slot = profiler->regions[region].open_slots[profiler->regions[region].depth - 1];
    }
    return scope;
}

int sp_scope_end(sp_scope *scope)
{
    sp_profiler *profiler;
    sp_region_t *r;

    if (scope == NULL || scope->slot < 0) {
        return SP_ERR_UNMATCHED_END;
    }
    profiler = scope->profiler;
    if (profiler == NULL || !profiler->active ||
        scope->region < 0 || scope->region >= profiler->n_regions) {
        scope->slot = -1;
        return SP_ERR_UNMATCHED_END;
    }
    r = &profiler->regions[scope->region];

    if (r->depth <= 0 || r->open_slots[r->depth - 1] != scope->slot) {
        /* Something else is on top -- ending this token now would mistime
         * whichever call actually is. The call this token names is still
         * open; leave the token valid so it can be ended once it is on top
         * (e.g. an inner scope, opened after this one, ends first). */
        profiler->last_error = SP_ERR_UNMATCHED_END;
        return SP_ERR_UNMATCHED_END;
    }

    if (profiler->open_depth > 0) {
        profiler->open_depth -= 1;
    }
    end_region(profiler, r);
    scope->slot = -1;
    return SP_OK;
}

int sp_profiler_reset(sp_profiler *profiler)
{
    int i;

    if (profiler == NULL) {
        return SP_ERR_INACTIVE;
    }
    for (i = 0; i < profiler->n_regions; ++i) {
        if (profiler->regions[i].depth != 0) {
            profiler->last_error = SP_ERR_OPEN_SCOPES;
            return SP_ERR_OPEN_SCOPES;
        }
    }
    for (i = 0; i < profiler->n_regions; ++i) {
        sp_region_t *r = &profiler->regions[i];

        r->ptr = 0;
        r->num_calls = 0;
        r->total_ns = 0;
        r->min_ns = 0;
        r->max_ns = 0;
        r->stat_calls = 0;
    }
    profiler->open_depth = 0;
    profiler->last_error = SP_OK;
    return SP_OK;
}

int sp_profiler_get_region_stats(
    const sp_profiler *profiler,
    int region,
    sp_region_stats *stats)
{
    const sp_region_t *r;

    if (profiler == NULL || stats == NULL || region < 0 || region >= profiler->n_regions) {
        return SP_ERR_INACTIVE;
    }
    r = &profiler->regions[region];
    stats->calls = r->num_calls;
    stats->total_ns = r->total_ns;
    stats->min_ns = r->min_ns;
    stats->max_ns = r->max_ns;
    return SP_OK;
}

/* "<prefix>_rank<NNNNN>.spt", cached on the profiler after the first call. */
static const char *trace_path(sp_profiler *profiler)
{
    const char *prefix;
    size_t length;

    if (profiler->cached_path != NULL) {
        return profiler->cached_path;
    }

    prefix = profiler->output_prefix != NULL ? profiler->output_prefix : "scope_profile";
    length = strlen(prefix) + 32;
    profiler->cached_path = (char *)malloc(length);
    if (profiler->cached_path == NULL) {
        return NULL;
    }
    snprintf(profiler->cached_path, length, "%s_rank%05d.spt", prefix, profiler->rank_id);
    return profiler->cached_path;
}

const char *sp_profiler_output_path(const sp_profiler *profiler)
{
    if (profiler == NULL) {
        return NULL;
    }
    /* trace_path() only mutates the cache, not anything observable; cast
     * away const to share the lazy-build logic with the writers below. */
    return trace_path((sp_profiler *)profiler);
}

/* How many of a region's reserved slots have both a start and an end time:
 * everything, unless a call is still open, in which case its slot (and any
 * nested ones after it) has no end time yet and is excluded. */
static int64_t written_count(const sp_region_t *region)
{
    return region->depth > 0 ? region->open_slots[0] : region->ptr;
}

/* Shared by sp_profiler_flush() and sp_profiler_finalize(). Does not modify
 * `profiler` -- flush must be safe to call while profiling continues. */
static int write_trace(sp_profiler *profiler)
{
    const char *path;
    FILE *out;
    int32_t version = SP_FORMAT_VERSION;
    int32_t rank = (int32_t)profiler->rank_id;
    int64_t written_regions = 0;
    int i;
    int failed = 0;

    for (i = 0; i < profiler->n_regions; ++i) {
        if (written_count(&profiler->regions[i]) > 0) {
            written_regions += 1;
        }
    }

    path = trace_path(profiler);
    if (path == NULL) {
        fprintf(stderr, "scope_profiler: out of memory writing the trace\n");
        profiler->last_error = SP_ERR_NO_MEMORY;
        return 1;
    }

    out = fopen(path, "wb");
    if (out == NULL) {
        fprintf(stderr, "scope_profiler: cannot write %s\n", path);
        profiler->last_error = SP_ERR_IO;
        return 1;
    }

    failed |= fwrite(SP_MAGIC, sizeof(SP_MAGIC), 1, out) != 1;
    failed |= fwrite(&version, sizeof(version), 1, out) != 1;
    failed |= fwrite(&rank, sizeof(rank), 1, out) != 1;
    failed |= fwrite(&written_regions, sizeof(written_regions), 1, out) != 1;

    for (i = 0; i < profiler->n_regions && !failed; ++i) {
        sp_region_t *region = &profiler->regions[i];
        int32_t name_len;
        int32_t source_len;
        int32_t source_line;
        int64_t count;

        count = written_count(region);
        if (count == 0) {
            continue;
        }
        name_len = (int32_t)strlen(region->name);
        source_len = region->source_file != NULL ? (int32_t)strlen(region->source_file) : 0;
        source_line = region->source_file != NULL ? region->source_line : -1;

        failed |= fwrite(&name_len, sizeof(name_len), 1, out) != 1;
        failed |= fwrite(region->name, 1, (size_t)name_len, out) != (size_t)name_len;
        failed |= fwrite(&source_len, sizeof(source_len), 1, out) != 1;
        failed |= source_len > 0 &&
                  fwrite(region->source_file, 1, (size_t)source_len, out) != (size_t)source_len;
        failed |= fwrite(&source_line, sizeof(source_line), 1, out) != 1;
        failed |= fwrite(&count, sizeof(count), 1, out) != 1;
        failed |= fwrite(region->start_times, sizeof(int64_t), (size_t)count, out) != (size_t)count;
        failed |= fwrite(region->end_times, sizeof(int64_t), (size_t)count, out) != (size_t)count;
    }

    if (fclose(out) != 0) {
        failed = 1;
    }
    if (failed) {
        fprintf(stderr, "scope_profiler: failed to write %s\n", path);
        profiler->last_error = SP_ERR_IO;
    }
    return failed;
}

int sp_profiler_flush(sp_profiler *profiler)
{
    if (profiler == NULL || !profiler->active) {
        return 0;
    }
    return write_trace(profiler);
}

int sp_profiler_finalize(sp_profiler *profiler)
{
    int i;
    int failed;

    if (profiler == NULL || !profiler->active) {
        return 0;
    }

    for (i = 0; i < profiler->n_regions; ++i) {
        if (profiler->regions[i].depth != 0) {
            fprintf(stderr,
                    "scope_profiler: region '%s' still open at sp_finalize "
                    "(depth %d); its last call(s) are dropped\n",
                    profiler->regions[i].name, profiler->regions[i].depth);
            profiler->last_error = SP_ERR_OPEN_SCOPES;
        }
    }

    failed = write_trace(profiler);

    profiler->active = 0;
    release_buffers(profiler);
    return failed;
}

/* ---------------------------------------------------------------------- */
/* Global convenience API: delegates to a hidden default sp_profiler.      */
/* ---------------------------------------------------------------------- */

int sp_init(const char *prefix, int rank)
{
    sp_destroy(g_default);
    g_default = sp_create(prefix, rank);
    if (g_default == NULL) {
        fprintf(stderr, "scope_profiler: out of memory in sp_init\n");
        return 1;
    }
    return g_default->active ? 0 : 1;
}

int sp_region(const char *name)
{
    return sp_profiler_region(g_default, name);
}

int sp_region_at(const char *name, const char *source_file, int source_line)
{
    return sp_profiler_region_at(g_default, name, source_file, source_line);
}

void sp_begin(int region)
{
    sp_profiler_begin(g_default, region);
}

void sp_end(int region)
{
    sp_profiler_end(g_default, region);
}

int64_t sp_num_calls(int region)
{
    sp_region_stats stats;

    if (sp_profiler_get_region_stats(g_default, region, &stats) != SP_OK) {
        return 0;
    }
    return stats.calls;
}

int sp_is_active(void)
{
    return sp_profiler_is_active(g_default);
}

int sp_finalize(void)
{
    return sp_profiler_finalize(g_default);
}

sp_profiler *sp_default_profiler(void)
{
    return g_default;
}
