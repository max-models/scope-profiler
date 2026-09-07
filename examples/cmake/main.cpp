#include <scope_profiler.hpp>

static void solve()
{
    SP_PROFILE_FUNCTION();
    volatile double result = 0.0;
    for (int i = 1; i < 10000; ++i) {
        result += 1.0 / i;
    }
}

int main()
{
    sp_init("cmake-profile", 0);
    solve();
    return sp_finalize();
}
