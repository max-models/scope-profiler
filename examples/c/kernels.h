/* A small C solver library that profiles its own internals.
 *
 * The same library is used two ways: `standalone.c` links it into a pure C
 * program, and `run_mixed.py` loads it with ctypes and drives it from Python.
 * Either way the regions inside it end up in the profile.
 *
 * Region names are prefixed `c:` so they cannot collide with the driver's own
 * regions -- scope-profiler refuses to merge a name recorded on both sides,
 * since that would double-count a wrapper and the region inside it.
 */
#ifndef KERNELS_H
#define KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Begin recording. Call once, before any kernel.
 *
 * prefix: output prefix; the trace lands in "<prefix>_rank<NNNNN>.spt"
 * rank:   MPI rank, so each process writes its own trace */
void kernels_start_profiling(const char *prefix, int rank);

/* Stop recording and write the trace. Call before the driver finalizes. */
void kernels_stop_profiling(void);

/* A Jacobi smoother on a 1-D grid, profiled sweep by sweep.
 *
 * Records two regions per call: the stencil update, and the residual
 * reduction that would decide whether to stop. Returns the final residual so
 * nothing is optimised away. */
double kernels_jacobi_solve(int n, int iterations);

/* Stand-in for writing a checkpoint: a region entered rarely.
 *
 * Rare regions are worth marking too -- they are what a Gantt chart makes
 * obvious and a total-time table hides. */
void kernels_checkpoint(int n);

#ifdef __cplusplus
}
#endif

#endif /* KERNELS_H */
