#include "kernels.h"

#include "scope_profiler.h"

#include <math.h>
#include <stdlib.h>

void kernels_start_profiling(const char *prefix, int rank)
{
    sp_init(prefix, rank);
}

void kernels_stop_profiling(void) { sp_finalize(); }

double kernels_jacobi_solve(int n, int iterations)
{
    double *u;
    double *u_new;
    double *swap;
    double residual = 0.0;
    int stencil_id;
    int residual_id;
    int sweep;
    int i;

    stencil_id = sp_region("c:stencil");
    residual_id = sp_region("c:residual");

    u = (double *)calloc((size_t)n, sizeof(double));
    u_new = (double *)calloc((size_t)n, sizeof(double));
    if (u == NULL || u_new == NULL) {
        free(u);
        free(u_new);
        return -1.0;
    }

    u[0] = 1.0;
    u_new[0] = 1.0;

    for (sweep = 0; sweep < iterations; ++sweep) {
        sp_begin(stencil_id);
        for (i = 1; i < n - 1; ++i) {
            u_new[i] = 0.5 * (u[i - 1] + u[i + 1]);
        }
        u_new[0] = u[0];
        u_new[n - 1] = u[n - 1];
        sp_end(stencil_id);

        sp_begin(residual_id);
        residual = 0.0;
        for (i = 1; i < n - 1; ++i) {
            residual += fabs(u_new[i] - u[i]);
        }
        /* Swap rather than copy, so the residual is the only O(n) read here. */
        swap = u;
        u = u_new;
        u_new = swap;
        sp_end(residual_id);
    }

    free(u);
    free(u_new);
    return residual;
}

void kernels_checkpoint(int n)
{
    int id = sp_region("c:checkpoint");
    double acc = 0.0;
    int i;

    sp_begin(id);
    for (i = 1; i <= n; ++i) {
        acc += sqrt((double)i);
    }
    /* Never taken; defeats the optimiser without printing anything. */
    if (acc < 0.0) {
        abort();
    }
    sp_end(id);
}
