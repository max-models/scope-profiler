/* Region profiling for C and C++, on the same timeline as scope-profiler.
 *
 * Records nanosecond start/end timestamps for named regions and writes them to
 * a trace file that `scope-profiler import-native` turns into the usual HDF5
 * output, so a C run gets the same summaries, plots and exports as a Python
 * one. The trace format is shared with the Fortran API, so a program built
 * from both lands in one profile.
 *
 * Self-contained C99: one .c file, no dependencies beyond libc, no HDF5, no
 * MPI. Safe to include from C++ (everything is extern "C").
 *
 * Two ways to use it:
 *
 * 1. Global convenience functions, for a single profiler per process:
 *
 *        #include "scope_profiler.h"
 *
 *        sp_init("profile", my_rank);          // rank 0 if you are serial
 *        int solve = sp_region("solve");       // resolve the name once
 *        for (int step = 0; step < nsteps; ++step) {
 *            sp_begin(solve);
 *            solve_system();
 *            sp_end(solve);
 *        }
 *        sp_finalize();                        // writes profile_rank00000.spt
 *
 *    These operate on a hidden default context, and are always safe to call
 *    even before sp_init() -- sp_region() returns SP_INVALID_REGION and
 *    sp_begin/sp_end silently ignore it, so instrumentation can stay in a
 *    build that never profiles.
 *
 * 2. Explicit `sp_profiler` contexts, for library code that must not
 *    interfere with a caller's own profiling (or with another instance of
 *    itself):
 *
 *        sp_profiler *p = sp_create("solver", my_rank);
 *        int solve = sp_profiler_region(p, "solve");
 *        sp_profiler_begin(p, solve);
 *        solve_system();
 *        sp_profiler_end(p, solve);
 *        sp_profiler_finalize(p);
 *        sp_destroy(p);
 *
 *    Every sp_profiler_*() function accepts a NULL or inactive profiler
 *    harmlessly, the same way the global functions tolerate calls before
 *    sp_init().
 *
 * Timestamps come from the same OS clock CPython's `time.perf_counter_ns()`
 * uses, so regions recorded here share an epoch with regions recorded by the
 * Python API in the same process, and land on one timeline.
 */
#ifndef SCOPE_PROFILER_H
#define SCOPE_PROFILER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Trace format version written for this profiler; keep in step with
 * native_trace.py. Bumped to 2 to add per-region source location; a version-2
 * reader still accepts version-1 (Fortran-written) files with no source
 * information. */
#define SP_FORMAT_VERSION 2

/* Returned by sp_region()/sp_region_at() when profiling is not running.
 * sp_begin/sp_end ignore it, so unprofiled builds need no conditionals at the
 * call sites. */
#define SP_INVALID_REGION (-1)

/* An opaque profiler instance; see sp_create(). */
typedef struct sp_profiler sp_profiler;

/* What went wrong in the most recent call against a profiler, readable with
 * sp_profiler_last_error() / sp_last_error(). */
typedef enum {
    SP_OK = 0,
    SP_ERR_INACTIVE,       /* the profiler is NULL, or not (yet, or still) active */
    SP_ERR_NO_CLOCK,       /* clock_gettime rejected every candidate clock */
    SP_ERR_NO_MEMORY,      /* malloc/realloc failed */
    SP_ERR_IO,             /* the trace file could not be written */
    SP_ERR_UNMATCHED_END,  /* sp_end()/sp_scope_end() with no open call to close */
    SP_ERR_OPEN_SCOPES     /* an operation refused because a region is still open */
} sp_status;

/* A human-readable name for a status, e.g. for logging. Never NULL. */
const char *sp_error_string(sp_status status);

/* Counters read back with sp_get_region_stats() / sp_profiler_get_region_stats(). */
typedef struct {
    int64_t calls;     /* number of completed sp_begin/sp_end pairs */
    int64_t total_ns;  /* sum of their durations */
    int64_t min_ns;    /* shortest duration seen (0 if calls == 0) */
    int64_t max_ns;    /* longest duration seen (0 if calls == 0) */
} sp_region_stats;

/* A single open call, returned by sp_scope_begin()/sp_profiler_scope_begin()
 * and consumed by sp_scope_end()/sp_profiler_scope_end(). Opaque: a C++ RAII
 * wrapper stores one of these per scope and passes it to sp_scope_end() in
 * its destructor. Ending it twice, or after a move, is harmless -- the second
 * call finds nothing open and returns SP_ERR_UNMATCHED_END rather than
 * corrupting another call's timing. */
typedef struct {
    sp_profiler *profiler;
    int region;
    int64_t slot;  /* internal; -1 marks a scope with nothing left to end */
} sp_scope;

/* ---------------------------------------------------------------------- */
/* Explicit-context API.                                                   */
/* ---------------------------------------------------------------------- */

/* Create a profiler that writes "<prefix>_rank<NNNNN>.spt" at
 * sp_profiler_finalize()/sp_profiler_flush().
 *
 * prefix: output path prefix.
 * rank:   MPI rank of this process, so each rank writes its own trace.
 *
 * Returns a new, empty, inactive-only-if-no-clock-was-found profiler; NULL
 * only if the sp_profiler struct itself could not be allocated. A profiler
 * returned because no clock was available is safe to use -- every call
 * against it is a harmless no-op -- and sp_profiler_last_error() reports
 * SP_ERR_NO_CLOCK. Never NULL to check for that case; check
 * sp_profiler_is_active() or sp_profiler_last_error() instead. */
sp_profiler *sp_create(const char *prefix, int rank);

/* Release a profiler and everything it owns. Does not write a trace -- call
 * sp_profiler_finalize() first if you want one. Safe to call with NULL. */
void sp_destroy(sp_profiler *profiler);

/* Handle for a region name, created on first use.
 *
 * Resolve once, outside hot loops, and pass the handle to sp_profiler_begin/
 * sp_profiler_end. Returns SP_INVALID_REGION if the profiler is NULL,
 * inactive, or memory ran out. */
int sp_profiler_region(sp_profiler *profiler, const char *name);

/* Like sp_profiler_region(), but also records where the region is defined.
 * source_file may be NULL if unknown; source_line is ignored when
 * source_file is NULL.
 *
 * A name registered for the first time without a source location (via
 * sp_profiler_region(), or sp_profiler_region_at() with a NULL source_file)
 * can still pick one up later: the next sp_profiler_region_at() call for the
 * same name backfills it onto the existing handle. Once a handle has a
 * source location, later calls never overwrite it -- first writer wins. */
int sp_profiler_region_at(
    sp_profiler *profiler,
    const char *name,
    const char *source_file,
    int source_line);

/* sp_profiler_region_at(profiler, name, ...) with source_file and
 * source_line filled in from the call site automatically. */
#define SP_PROFILER_REGION_AT(profiler, name) \
    sp_profiler_region_at((profiler), (name), __FILE__, __LINE__)

/* Enter a region. Reserves this call's slot before the work starts, so a
 * recursive re-entry cannot overwrite it. */
void sp_profiler_begin(sp_profiler *profiler, int region);

/* Leave a region, writing the end time into the slot sp_profiler_begin
 * reserved. Sets SP_ERR_UNMATCHED_END if this region has no open call. */
void sp_profiler_end(sp_profiler *profiler, int region);

/* Leave whichever region was most recently entered (and not yet left) on this
 * profiler, regardless of which region that is. Supports call sites that
 * track "the current section" themselves instead of naming it again to close
 * it. Returns SP_ERR_UNMATCHED_END if nothing is open. */
int sp_profiler_end_last(sp_profiler *profiler);

/* Enter a region and return a token for exactly this call. */
sp_scope sp_profiler_scope_begin(sp_profiler *profiler, int region);

/* Leave the call `scope` was returned for.
 *
 * Checked: if another call on the same region was opened after this one and
 * has not yet been closed, ending it here is refused (SP_ERR_UNMATCHED_END)
 * instead of mistiming whichever call is actually on top -- the token stays
 * valid, so ending it again later (once it is on top, e.g. after an inner
 * scope opened after it has ended) succeeds normally. If this scope was
 * already ended, ending it again is also refused, but harmlessly: it is left
 * inert. Returns an sp_status, as an int for a stable C ABI. */
int sp_scope_end(sp_scope *scope);

/* Discard every recorded call and reset each region's statistics, but keep
 * every region name and handle valid -- resolve them again is unnecessary.
 * For periodic reporting followed by a fresh measurement window.
 *
 * Refuses (SP_ERR_OPEN_SCOPES) if any region has a call still open; finish or
 * abandon it first. */
int sp_profiler_reset(sp_profiler *profiler);

/* Read a region's call count and duration statistics. Returns SP_OK, or
 * SP_ERR_INACTIVE if profiler/stats is NULL or region is out of range. */
int sp_profiler_get_region_stats(
    const sp_profiler *profiler,
    int region,
    sp_region_stats *stats);

/* Number of distinct regions registered so far (0 if profiler is NULL). */
int sp_profiler_num_regions(const sp_profiler *profiler);

/* The name a region was registered with, or NULL if region is out of range. */
const char *sp_profiler_region_name(const sp_profiler *profiler, int region);

/* Write everything recorded so far to "<prefix>_rank<NNNNN>.spt", without
 * stopping profiling or discarding anything -- calls still open are simply
 * not yet in the file. For long-running applications that want to publish
 * completed data before the run ends. Returns 0 on success, non-zero if the
 * trace could not be written (also recorded as SP_ERR_IO). */
int sp_profiler_flush(sp_profiler *profiler);

/* Write this profiler's trace, release its recording buffers, and stop
 * profiling. Region names and statistics remain readable afterwards.
 *
 * A region still open is reported on stderr and its unterminated call
 * dropped, rather than written with a missing end time (also recorded as
 * SP_ERR_OPEN_SCOPES).
 *
 * Returns 0 on success, non-zero if the trace could not be written. */
int sp_profiler_finalize(sp_profiler *profiler);

/* Whether this profiler is running (created with a clock and not yet
 * finalized). False for NULL. */
int sp_profiler_is_active(const sp_profiler *profiler);

/* The path sp_profiler_finalize()/sp_profiler_flush() writes (or wrote), or
 * NULL for a NULL profiler. */
const char *sp_profiler_output_path(const sp_profiler *profiler);

/* The status of the most recent call against this profiler. SP_OK for NULL
 * (nothing has gone wrong on behalf of a profiler that does not exist). */
sp_status sp_profiler_last_error(const sp_profiler *profiler);

/* ---------------------------------------------------------------------- */
/* Global convenience API: a hidden default sp_profiler, one per process.  */
/* ---------------------------------------------------------------------- */

/* Start profiling on the default context. Call once, before any other global
 * function below. A second call replaces the previous default context (as if
 * sp_destroy() had been called on it).
 *
 * Returns 0 on success, non-zero if no monotonic clock is available (in which
 * case profiling stays off rather than recording meaningless timestamps). */
int sp_init(const char *prefix, int rank);

/* sp_profiler_region() against the default context. */
int sp_region(const char *name);

/* sp_profiler_region_at() against the default context. */
int sp_region_at(const char *name, const char *source_file, int source_line);

/* sp_region_at(name, ...) with source_file and source_line filled in from
 * the call site automatically. */
#define SP_REGION_AT(name) sp_region_at((name), __FILE__, __LINE__)

/* sp_profiler_begin() against the default context. */
void sp_begin(int region);

/* sp_profiler_end() against the default context. */
void sp_end(int region);

/* Number of times a region was entered. Readable after sp_finalize(). */
int64_t sp_num_calls(int region);

/* Nanoseconds on the same clock as Python's time.perf_counter_ns().
 * Negative if no clock could be resolved. Shared by every profiler. */
int64_t sp_now_ns(void);

/* Whether the default context is running (sp_init succeeded, sp_finalize not
 * yet called). */
int sp_is_active(void);

/* sp_profiler_finalize() against the default context. */
int sp_finalize(void);

/* The profiler behind every function above. NULL until sp_init() has been
 * called at least once; non-NULL (though possibly inactive, e.g. after
 * sp_finalize()) from then on, even across a later sp_finalize().
 *
 * Exists so wrapper code (e.g. scope_profiler.hpp's RAII class) can use the
 * checked explicit-context primitives, such as sp_profiler_scope_begin(),
 * against the same default context sp_region()/sp_begin()/sp_end() use,
 * instead of duplicating that default. */
sp_profiler *sp_default_profiler(void);

#ifdef __cplusplus
}
#endif

#endif /* SCOPE_PROFILER_H */
