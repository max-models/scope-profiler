/* A C++ RAII wrapper around the checked scope-token API in scope_profiler.h.
 *
 * Header-only wrapper, C++11, with no dependencies beyond scope_profiler.h.
 * Link scope_profiler.c normally, or define SP_HEADER_ONLY under C++17 to
 * include inline definitions and require no separately compiled source. Use
 * it from translation units that want automatic sp_scope_end() on every exit
 * path (return, break,
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

/* The inline backend needs the clock declarations before scope_profiler.h
 * includes the first system header. Applications may also define their own
 * compatible feature-test level before including this file. */
#ifdef SP_HEADER_ONLY
#  if defined(__APPLE__)
#    ifndef _DARWIN_C_SOURCE
#      define _DARWIN_C_SOURCE
#    endif
#  elif !defined(_POSIX_C_SOURCE)
#    define _POSIX_C_SOURCE 200809L
#  endif
#endif

#include "scope_profiler.h"

#if defined(SP_USE_LIKWID) && !defined(SP_DISABLE_PROFILING)
#  include <likwid-marker.h>
#endif

namespace sp {

class Scope {
public:
    /* Enter `region` on `profiler` now; leave it when this Scope is
     * destroyed (or moved from). */
    Scope(sp_profiler *profiler, int region) noexcept
        : scope_(sp_profiler_scope_begin(profiler, region))
#if defined(SP_USE_LIKWID) && !defined(SP_DISABLE_PROFILING)
          , counter_name_(sp_profiler_region_name(profiler, region)),
          counter_active_(counter_name_ != 0)
#endif
    {
        counter_start();
    }

    /* Enter `region` on the default context -- the one sp_init()/sp_region()/
     * sp_begin()/sp_end() operate on. */
    explicit Scope(int region) noexcept
        : scope_(sp_profiler_scope_begin(sp_default_profiler(), region))
#if defined(SP_USE_LIKWID) && !defined(SP_DISABLE_PROFILING)
          , counter_name_(sp_profiler_region_name(sp_default_profiler(), region)),
          counter_active_(counter_name_ != 0)
#endif
    {
        counter_start();
    }

    ~Scope()
    {
        counter_stop();
        sp_scope_end(&scope_);
    }

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

    Scope(Scope &&other) noexcept
        : scope_(other.scope_)
#if defined(SP_USE_LIKWID) && !defined(SP_DISABLE_PROFILING)
          , counter_name_(other.counter_name_),
          counter_active_(other.counter_active_)
#endif
    {
        other.scope_.slot = -1;
#if defined(SP_USE_LIKWID) && !defined(SP_DISABLE_PROFILING)
        other.counter_active_ = false;
#endif
    }

    Scope &operator=(Scope &&other) noexcept
    {
        if (this != &other) {
            counter_stop();
            sp_scope_end(&scope_);
            scope_ = other.scope_;
            other.scope_.slot = -1;
#if defined(SP_USE_LIKWID) && !defined(SP_DISABLE_PROFILING)
            counter_name_ = other.counter_name_;
            counter_active_ = other.counter_active_;
            other.counter_active_ = false;
#endif
        }
        return *this;
    }

private:
    void counter_start() noexcept
    {
#if defined(SP_USE_LIKWID) && !defined(SP_DISABLE_PROFILING)
        if (counter_active_) {
            LIKWID_MARKER_START(counter_name_);
        }
#endif
    }

    void counter_stop() noexcept
    {
#if defined(SP_USE_LIKWID) && !defined(SP_DISABLE_PROFILING)
        if (counter_active_) {
            LIKWID_MARKER_STOP(counter_name_);
            counter_active_ = false;
        }
#endif
    }

    sp_scope scope_;
#if defined(SP_USE_LIKWID) && !defined(SP_DISABLE_PROFILING)
    const char *counter_name_;
    bool counter_active_;
#endif
};

/* Own LIKWID's process-wide marker session when native counter collection is
 * enabled. LIKWID itself decides whether counters are active (the executable
 * normally runs under `likwid-perfctr -m`). */
class LikwidSession {
public:
    LikwidSession() noexcept
    {
#if defined(SP_USE_LIKWID) && !defined(SP_DISABLE_PROFILING)
        LIKWID_MARKER_INIT;
        LIKWID_MARKER_THREADINIT;
#endif
    }

    ~LikwidSession()
    {
#if defined(SP_USE_LIKWID) && !defined(SP_DISABLE_PROFILING)
        LIKWID_MARKER_CLOSE;
#endif
    }

    LikwidSession(const LikwidSession &) = delete;
    LikwidSession &operator=(const LikwidSession &) = delete;
};

/* Register a worker thread with LIKWID before it enters marker scopes. It is
 * an inline no-op unless native LIKWID support is enabled. */
inline void likwid_thread_init() noexcept
{
#if defined(SP_USE_LIKWID) && !defined(SP_DISABLE_PROFILING)
    LIKWID_MARKER_THREADINIT;
#endif
}

/* The object emitted by SP_PROFILE_SCOPE*. It always records the native
 * timeline and, with SP_USE_LIKWID, wraps that same named region in LIKWID's
 * marker API. */
class ProfileScope {
public:
    ProfileScope(const char *name, const char *file, int line) noexcept
        : scope_(sp_default_profiler(), sp_region_at(name, file, line))
    {
    }

    ProfileScope(
        sp_profiler *profiler,
        const char *name,
        const char *file,
        int line) noexcept
        : scope_(profiler, sp_profiler_region_at(profiler, name, file, line))
    {
    }

    ~ProfileScope() = default;

    ProfileScope(const ProfileScope &) = delete;
    ProfileScope &operator=(const ProfileScope &) = delete;
    ProfileScope(ProfileScope &&) = delete;
    ProfileScope &operator=(ProfileScope &&) = delete;

private:
    Scope scope_;
};

} // namespace sp

#define SP_DETAIL_JOIN_INNER(a, b) a##b
#define SP_DETAIL_JOIN(a, b) SP_DETAIL_JOIN_INNER(a, b)

#if defined(_MSC_VER)
#  define SP_DETAIL_FUNCTION_NAME __FUNCSIG__
#elif defined(__GNUC__) || defined(__clang__)
#  define SP_DETAIL_FUNCTION_NAME __PRETTY_FUNCTION__
#else
#  define SP_DETAIL_FUNCTION_NAME __func__
#endif

/* These are deliberately the only recommended call-site macros. With
 * SP_DISABLE_PROFILING they do not evaluate their arguments, retain region
 * strings, instantiate objects, or reference the profiler ABI. */
#ifdef SP_DISABLE_PROFILING
#  define SP_PROFILE_SCOPE(name) ((void)0)
#  define SP_PROFILE_SCOPE_CTX(profiler, name) ((void)0)
#  define SP_PROFILE_FUNCTION() ((void)0)
#  define SP_PROFILE_FUNCTION_CTX(profiler) ((void)0)
#else
#  define SP_PROFILE_SCOPE(name) \
    ::sp::ProfileScope SP_DETAIL_JOIN(sp_profile_scope_, __LINE__)( \
        (name), __FILE__, __LINE__)
#  define SP_PROFILE_SCOPE_CTX(profiler, name) \
    ::sp::ProfileScope SP_DETAIL_JOIN(sp_profile_scope_, __LINE__)( \
        (profiler), (name), __FILE__, __LINE__)
#  define SP_PROFILE_FUNCTION() SP_PROFILE_SCOPE(SP_DETAIL_FUNCTION_NAME)
#  define SP_PROFILE_FUNCTION_CTX(profiler) \
    SP_PROFILE_SCOPE_CTX((profiler), SP_DETAIL_FUNCTION_NAME)
#endif

/* Include last: the declarations and C++ wrappers above must already be
 * visible, and the implementation's feature-test macros must precede its own
 * system headers. */
#ifdef SP_HEADER_ONLY
#  include "scope_profiler_impl.h"
#endif

#endif /* SCOPE_PROFILER_HPP */
