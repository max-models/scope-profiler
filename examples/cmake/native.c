#include <scope_profiler.h>

int main(void)
{
    int solve;
    volatile double result = 0.0;
    int i;

    sp_init("cmake-profile", 0);
    solve = sp_region("solve");
    sp_begin(solve);
    for (i = 1; i < 10000; ++i) {
        result += 1.0 / i;
    }
    sp_end(solve);
    return sp_finalize();
}
