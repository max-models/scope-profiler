/* A C++ RAII wrapper around the checked scope-token API in scope_profiler.h.
 *
 * Header-only, C++11, no dependencies beyond scope_profiler.h itself -- drop
 * it next to scope_profiler.c/.h and #include it from C++ translation units
 * that want automatic sp_scope_end() on every exit path (return, break,
 * exception) instead of a hand-written sp_begin()/sp_end() pair, or a
 * hand-written wrapper of one's own.
 *
 *     #include "scope_profiler.hpp"
 *
 *     sp_profiler *p = sp_create("solver", rank);
 *     int solve = sp_profiler_region(p, "solve");
 *
 *     void step()
 *     {
 *         sp::Scope timed(p, solve);
 *         solve_system();               // sp_scope_end() runs however this
 *                                        // function exits, including a throw
 *     }
 *
 * Or, against the default context sp_init()/sp_region() set up:
 *
 *     sp_init("profile", rank);
 *     int solve = sp_region("solve");
 *     ...
 *     sp::Scope timed(solve);           // no profiler pointer needed
 *
 * sp::Scope is move-only, matching the token it wraps: a moved-from Scope
 * ends nothing at its own destruction, and ending the same call twice
 * (through two different Scope objects, say) is harmless -- the second finds
 * nothing left to end.
 */
#ifndef SCOPE_PROFILER_HPP
#define SCOPE_PROFILER_HPP

#include "scope_profiler.h"

namespace sp {

class Scope {
public:
    /* Enter `region` on `profiler` now; leave it when this Scope is
     * destroyed (or moved from). */
    Scope(sp_profiler *profiler, int region) noexcept
        : scope_(sp_profiler_scope_begin(profiler, region))
    {
    }

    /* Enter `region` on the default context -- the one sp_init()/sp_region()/
     * sp_begin()/sp_end() operate on. */
    explicit Scope(int region) noexcept
        : scope_(sp_profiler_scope_begin(sp_default_profiler(), region))
    {
    }

    ~Scope() { sp_scope_end(&scope_); }

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

    Scope(Scope &&other) noexcept : scope_(other.scope_) { other.scope_.slot = -1; }

    Scope &operator=(Scope &&other) noexcept
    {
        if (this != &other) {
            sp_scope_end(&scope_);
            scope_ = other.scope_;
            other.scope_.slot = -1;
        }
        return *this;
    }

private:
    sp_scope scope_;
};

} // namespace sp

#endif /* SCOPE_PROFILER_HPP */
