/* Runnable example of the C region API.
 *
 *     cc -O2 -c scope_profiler.c
 *     cc -O2 example.c scope_profiler.o -lm -o example
 *     ./example
 *     scope-profiler import-native . -o profiling_data.h5
 *     scope-profiler inspect profiling_data.h5
 */
#include "scope_profiler.h"

#include <math.h>
#include <stdio.h>

/* Something the compiler cannot optimise away. */
static double busy_work(int n)
{
    double acc = 0.0;
    int i;

    for (i = 1; i <= n; ++i) {
        acc += sqrt((double)i);
    }
    return acc;
}

/* Recursion: every invocation reserves its own slot in the same region. */
static int fib(int n)
{
    int id = sp_region("fib_call");
    int value;

    sp_begin(id);
    value = n < 2 ? n : fib(n - 1) + fib(n - 2);
    sp_end(id);
    return value;
}

int main(void)
{
    int step, solve, assemble, io;
    double total = 0.0;

    sp_init("profile", 0);

    /* Resolve the names once; the handles are what the hot loop uses. */
    solve = sp_region("solve");
    assemble = sp_region("assemble");
    io = sp_region("checkpoint");

    for (step = 1; step <= 20; ++step) {
        sp_begin(assemble);
        total += busy_work(20000);
        sp_end(assemble);

        sp_begin(solve);
        total += busy_work(50000);
        sp_end(solve);

        if (step % 10 == 0) {
            sp_begin(io);
            total += busy_work(5000);
            sp_end(io);
        }
    }

    sp_begin(sp_region("fibonacci"));
    step = fib(12);
    sp_end(sp_region("fibonacci"));

    printf("solve entered %lld time(s)\n", (long long)sp_num_calls(solve));
    printf("checksum (ignore): %.4f\n", total + (double)step);

    if (sp_finalize() != 0) {
        return 1;
    }
    printf("wrote profile_rank00000.spt\n");
    return 0;
}
