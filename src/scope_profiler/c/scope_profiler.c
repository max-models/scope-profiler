/* Implementation of the C region API; see scope_profiler.h.
 *
 * The trace format is the one the Fortran API writes, documented in
 * scope_profiler/native_trace.py:
 *
 *     char[8]   "SCOPEPRF"
 *     int32     format version
 *     int32     rank
 *     int64     number of regions
 *     per region:
 *         int32     length of the name in bytes
 *         char[]    name
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

typedef struct {
    char *name;
    int64_t *start_times;
    int64_t *end_times;
    int64_t ptr;      /* slots used */
    int64_t capacity;
    int64_t num_calls;
    /* Slots reserved by calls still open, innermost last, so a recursive
     * re-entry reserves its own slot instead of overwriting the outer one. */
    int64_t open_slots[SP_MAX_DEPTH];
    int depth;
} sp_region_t;

static sp_region_t *regions = NULL;
static int n_regions = 0;
static int regions_capacity = 0;
static int rank_id = 0;
static char *output_prefix = NULL;
static int active = 0;

/* Resolved on first use; -1 means "not yet probed", -2 "none works".
 *
 * CLOCK_MONOTONIC is what CPython's perf_counter_ns() reads on Linux;
 * CLOCK_UPTIME_RAW is what it reads on macOS. Sharing that clock is what puts
 * C regions and Python regions on one timeline. */
static clockid_t clock_id = (clockid_t)-1;
static int clock_resolved = 0;

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

int sp_is_active(void) { return active; }

/* Release the timestamp buffers but keep each region's name and call count.
 *
 * sp_finalize() uses this: the counts stay readable afterwards, matching the
 * Fortran and Python APIs, where a region's call count outlives the run it was
 * recorded in. The remaining allocation is bounded by the number of regions,
 * and sp_init() reclaims it. */
static void release_buffers(void)
{
    int i;

    for (i = 0; i < n_regions; ++i) {
        free(regions[i].start_times);
        free(regions[i].end_times);
        regions[i].start_times = NULL;
        regions[i].end_times = NULL;
        regions[i].capacity = 0;
        regions[i].ptr = 0;
        regions[i].depth = 0;
    }
}

static void release_all(void)
{
    int i;

    for (i = 0; i < n_regions; ++i) {
        free(regions[i].name);
        free(regions[i].start_times);
        free(regions[i].end_times);
    }
    free(regions);
    regions = NULL;
    n_regions = 0;
    regions_capacity = 0;

    free(output_prefix);
    output_prefix = NULL;
}

int sp_init(const char *prefix, int rank)
{
    resolve_clock();
    if (clock_id == (clockid_t)-2) {
        fprintf(stderr,
                "scope_profiler: no monotonic clock available "
                "(clock_gettime rejected every candidate); profiling is disabled\n");
        active = 0;
        return 1;
    }

    release_all();

    output_prefix = strdup(prefix != NULL ? prefix : "scope_profile");
    if (output_prefix == NULL) {
        fprintf(stderr, "scope_profiler: out of memory in sp_init\n");
        return 1;
    }

    rank_id = rank;
    active = 1;
    return 0;
}

int sp_region(const char *name)
{
    sp_region_t *bigger;
    sp_region_t *region;
    int i;

    if (!active || name == NULL) {
        return SP_INVALID_REGION;
    }

    for (i = 0; i < n_regions; ++i) {
        if (strcmp(regions[i].name, name) == 0) {
            return i;
        }
    }

    if (n_regions == regions_capacity) {
        int capacity = regions_capacity == 0 ? 16 : regions_capacity * 2;
        bigger = (sp_region_t *)realloc(regions, (size_t)capacity * sizeof(*regions));
        if (bigger == NULL) {
            fprintf(stderr, "scope_profiler: out of memory registering '%s'\n", name);
            return SP_INVALID_REGION;
        }
        regions = bigger;
        regions_capacity = capacity;
    }

    region = &regions[n_regions];
    memset(region, 0, sizeof(*region));
    region->name = strdup(name);
    region->start_times = (int64_t *)malloc(SP_INITIAL_CAPACITY * sizeof(int64_t));
    region->end_times = (int64_t *)malloc(SP_INITIAL_CAPACITY * sizeof(int64_t));
    if (region->name == NULL || region->start_times == NULL ||
        region->end_times == NULL) {
        free(region->name);
        free(region->start_times);
        free(region->end_times);
        fprintf(stderr, "scope_profiler: out of memory registering '%s'\n", name);
        return SP_INVALID_REGION;
    }
    region->capacity = SP_INITIAL_CAPACITY;

    return n_regions++;
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

void sp_begin(int id)
{
    sp_region_t *region;

    if (!active || id < 0 || id >= n_regions) {
        return;
    }
    region = &regions[id];

    if (region->ptr >= region->capacity && grow(region) != 0) {
        fprintf(stderr, "scope_profiler: out of memory recording '%s'; "
                        "this call is not timed\n",
                region->name);
        return;
    }
    region->num_calls += 1;

    if (region->depth >= SP_MAX_DEPTH) {
        fprintf(stderr,
                "scope_profiler: region '%s' nested deeper than %d; "
                "this call is not timed\n",
                region->name, SP_MAX_DEPTH);
        return;
    }

    region->open_slots[region->depth] = region->ptr;
    region->depth += 1;
    region->ptr += 1;
    region->start_times[region->open_slots[region->depth - 1]] = sp_now_ns();
}

void sp_end(int id)
{
    sp_region_t *region;
    int64_t slot;

    if (!active || id < 0 || id >= n_regions) {
        return;
    }
    region = &regions[id];

    if (region->depth <= 0) {
        fprintf(stderr, "scope_profiler: sp_end('%s') without a matching sp_begin\n",
                region->name);
        return;
    }

    region->depth -= 1;
    slot = region->open_slots[region->depth];
    region->end_times[slot] = sp_now_ns();
}

int64_t sp_num_calls(int id)
{
    if (id < 0 || id >= n_regions) {
        return 0;
    }
    return regions[id].num_calls;
}

/* "<prefix>_rank<NNNNN>.spt"; caller frees. NULL if out of memory. */
static char *trace_path(void)
{
    const char *prefix = output_prefix != NULL ? output_prefix : "scope_profile";
    size_t length = strlen(prefix) + 32;
    char *path = (char *)malloc(length);

    if (path == NULL) {
        return NULL;
    }
    snprintf(path, length, "%s_rank%05d.spt", prefix, rank_id);
    return path;
}

int sp_finalize(void)
{
    char *path;
    FILE *out;
    int32_t version = SP_FORMAT_VERSION;
    int32_t rank = (int32_t)rank_id;
    int64_t written_regions = 0;
    int i;
    int failed = 0;

    if (!active) {
        return 0;
    }

    for (i = 0; i < n_regions; ++i) {
        if (regions[i].depth != 0) {
            fprintf(stderr,
                    "scope_profiler: region '%s' still open at sp_finalize "
                    "(depth %d); its last call(s) are dropped\n",
                    regions[i].name, regions[i].depth);
            /* The reserved slots have no end time; do not write them out. */
            regions[i].ptr = regions[i].open_slots[0];
        }
        if (regions[i].ptr > 0) {
            written_regions += 1;
        }
    }

    path = trace_path();
    if (path == NULL) {
        fprintf(stderr, "scope_profiler: out of memory in sp_finalize\n");
        active = 0;
        release_buffers();
        return 1;
    }

    out = fopen(path, "wb");
    if (out == NULL) {
        fprintf(stderr, "scope_profiler: cannot write %s\n", path);
        free(path);
        active = 0;
        release_buffers();
        return 1;
    }

    failed |= fwrite(SP_MAGIC, sizeof(SP_MAGIC), 1, out) != 1;
    failed |= fwrite(&version, sizeof(version), 1, out) != 1;
    failed |= fwrite(&rank, sizeof(rank), 1, out) != 1;
    failed |= fwrite(&written_regions, sizeof(written_regions), 1, out) != 1;

    for (i = 0; i < n_regions && !failed; ++i) {
        sp_region_t *region = &regions[i];
        int32_t name_len;
        size_t count;

        if (region->ptr <= 0) {
            continue;
        }
        name_len = (int32_t)strlen(region->name);
        count = (size_t)region->ptr;

        failed |= fwrite(&name_len, sizeof(name_len), 1, out) != 1;
        failed |= fwrite(region->name, 1, (size_t)name_len, out) != (size_t)name_len;
        failed |= fwrite(&region->ptr, sizeof(region->ptr), 1, out) != 1;
        failed |= fwrite(region->start_times, sizeof(int64_t), count, out) != count;
        failed |= fwrite(region->end_times, sizeof(int64_t), count, out) != count;
    }

    if (fclose(out) != 0) {
        failed = 1;
    }
    if (failed) {
        fprintf(stderr, "scope_profiler: failed to write %s\n", path);
    }

    free(path);
    active = 0;
    release_buffers();
    return failed;
}
