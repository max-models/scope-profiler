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
 *     #include "scope_profiler.h"
 *
 *     sp_init("profile", my_rank);          // rank 0 if you are serial
 *     int solve = sp_region("solve");       // resolve the name once
 *     for (int step = 0; step < nsteps; ++step) {
 *         sp_begin(solve);
 *         solve_system();
 *         sp_end(solve);
 *     }
 *     sp_finalize();                        // writes profile_rank00000.spt
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

/* Trace format written by sp_finalize; keep in step with native_trace.py. */
#define SP_FORMAT_VERSION 1

/* Returned by sp_region() when profiling is not running. sp_begin/sp_end
 * ignore it, so unprofiled builds need no conditionals at the call sites. */
#define SP_INVALID_REGION (-1)

/* Start profiling. Call once, before any other function.
 *
 * prefix: output path prefix; sp_finalize writes "<prefix>_rank<NNNNN>.spt".
 * rank:   MPI rank of this process, so each rank writes its own trace.
 *
 * Returns 0 on success, non-zero if no monotonic clock is available (in which
 * case profiling stays off rather than recording meaningless timestamps). */
int sp_init(const char *prefix, int rank);

/* Handle for a region name, created on first use.
 *
 * Resolve once, outside hot loops, and pass the handle to sp_begin/sp_end.
 * Returns SP_INVALID_REGION if profiling is off or memory ran out. */
int sp_region(const char *name);

/* Enter a region. Reserves this call's slot before the work starts, so a
 * recursive re-entry cannot overwrite it. */
void sp_begin(int region);

/* Leave a region, writing the end time into the slot sp_begin reserved. */
void sp_end(int region);

/* Number of times a region was entered. Readable after sp_finalize(). */
int64_t sp_num_calls(int region);

/* Nanoseconds on the same clock as Python's time.perf_counter_ns().
 * Negative if no clock could be resolved. */
int64_t sp_now_ns(void);

/* Whether profiling is running (sp_init succeeded, sp_finalize not yet
 * called). */
int sp_is_active(void);

/* Write this rank's trace, release everything, and stop profiling.
 *
 * Regions never entered are skipped. A region still open is reported on
 * stderr and its unterminated call dropped, rather than written with a
 * missing end time.
 *
 * Returns 0 on success, non-zero if the trace could not be written. */
int sp_finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* SCOPE_PROFILER_H */
