/* A pure C program, profiled with no Python at run time.
 *
 *     make run-standalone
 *
 * Marks its own regions around the kernels' regions, so the resulting profile
 * shows both levels. Takes an optional rank argument, which is how you would
 * pass MPI_Comm_rank under a launcher:
 *
 *     mpirun -n 4 ./standalone   # if you build it with mpicc; see README.md
 */
#include "kernels.h"
#include "scope_profiler.h"

#include <stdio.h>
#include <stdlib.h>

#define GRID 20000
#define STEPS 20
#define SWEEPS 5

int main(int argc, char **argv)
{
    int rank = argc > 1 ? atoi(argv[1]) : 0;
    int step;
    int setup_id;
    int timestep_id;
    double residual = 0.0;

    kernels_start_profiling("standalone", rank);

    setup_id = sp_region("c:setup");
    timestep_id = sp_region("c:timestep");

    sp_begin(setup_id);
    kernels_checkpoint(50000); /* pretend: read input, build the grid */
    sp_end(setup_id);

    for (step = 1; step <= STEPS; ++step) {
        sp_begin(timestep_id);
        residual = kernels_jacobi_solve(GRID, SWEEPS);
        sp_end(timestep_id);

        /* Checkpoint every fifth step: a rare region among frequent ones. */
        if (step % 5 == 0) {
            kernels_checkpoint(200000);
        }
    }

    /* Read the output path before finalizing: it names whichever format
     * this build writes -- .h5 where the profiler was compiled with HDF5
     * (`make SP_HDF5=1`), the .spt trace otherwise. */
    printf("rank %d: done\n", rank);
    printf("  final residual: %.5e\n", residual);
    printf("  wrote %s\n", sp_output_path());
    if (sp_current_output_format() == SP_OUTPUT_TRACE) {
        printf("  now run: scope-profiler import-native . -o profiling_data.h5\n");
    } else {
        printf("  now run: scope-profiler inspect %s\n", sp_output_path());
    }
    kernels_stop_profiling();
    return 0;
}
