

# Agent optimization loop: a worked example

This directory is a minimal, runnable instance of the loop described in
the top-level `CLAUDE.md` under *Where it is going: agent-driven
benchmarking*:

> measure, change the code, measure again, and keep only the changes
> that are both faster and still correct.

It exists so an AI coding agent (or you) can run the loop once, end to
end, against a script small enough to read in one sitting, using nothing
but the scope-profiler MCP tools (`run_profile`, `compare_profiles`)
plus a plain pytest correctness gate.

## The pieces

- **`optimize_me.py`** — a small numerical pipeline (`smooth` →
  `transform` → `norm`) written as unvectorized pure-Python loops on
  purpose. It imports nothing from `scope_profiler`: run it under
  `scope-profiler run` (or the `run_profile` MCP tool) and every
  function call is auto-instrumented as its own region, with zero
  decorators.
- **`test_correctness.py`** — the correctness gate. It pins
  `run_pipeline(n=4000, seed=0)` to a reference value computed from the
  original implementation. Any “optimization” that changes this value is
  not a valid change, no matter how much faster it is.

## The loop, as an agent should run it

1.  **Measure a baseline.**

    ``` python
    run_profile("examples/agent_workflow/optimize_me.py",
                output_path="/tmp/baseline.h5")
    ```

    This is `inspect_profile`-shaped: per-region
    `total`/`avg`/`min`/`max` plus `total_time_seconds`. Read it to find
    where the time actually goes before touching any code — in this
    script, `smooth` and `transform` dominate; `norm` is comparatively
    cheap.

2.  **Change the code.** This step is outside scope-profiler’s scope by
    design (see `CLAUDE.md`) — the agent edits `optimize_me.py` using
    its normal tools. The obvious move here is vectorizing the
    pure-Python loops with numpy.

3.  **Re-measure.**

    ``` python
    run_profile("examples/agent_workflow/optimize_me.py",
                output_path="/tmp/candidate.h5")
    ```

4.  **Judge — did it get faster, and is it still correct?**

    ``` python
    compare_profiles("/tmp/baseline.h5", "/tmp/candidate.h5")
    ```

    Read `overall.faster` and `overall.speedup` directly — never
    subtract two `inspect_profile` calls by hand. Then run the
    correctness gate:

    ``` bash
    pytest examples/agent_workflow/test_correctness.py
    ```

    Keep the change only if **both** hold: `overall.faster is True`
    **and** the pytest run is green. If either fails, revert and try
    something else.

5.  **Repeat** from step 2 for the next hotspot, using the *candidate’s*
    own region breakdown to pick the next target — not the baseline’s,
    since the ranking of what’s expensive can change after each edit.

## What actually happened when this was run once

Vectorizing `smooth`, `transform` and `norm` with numpy (pure-Python
loops → numpy array ops) and re-running the loop above gave:

``` json
{
  "absolute_diff_seconds": -0.0148,
  "relative_change_pct": -44.1,
  "speedup": 1.79,
  "faster": true
}
```

`pytest test_correctness.py` stayed green — same `run_pipeline` output
to 9 significant figures, only the runtime changed.

The per-region breakdown is the more interesting part, and it’s the
reason `compare_profiles` reports both `overall` *and* per-region deltas
rather than just one number:

| region      | change                  |
|-------------|-------------------------|
| `transform` | **-99.0%**              |
| `smooth`    | **-95.4%**              |
| `norm`      | **+64.0%** (regression) |

`norm` got *slower* in isolation — at n=4000, the fixed overhead of a
numpy call (`np.linalg.norm`) outweighs what it saves over a tight
Python loop summing floats. An agent that only checked `overall.faster`
would still correctly keep this change (the pipeline is 1.79x faster end
to end and still correct), but one that iterates region-by-region needs
to know not to chase the `norm` regression next — it’s real, but it’s
not where the time is. This is exactly the kind of judgment `overall` +
`regressions` + `improvements` in `compare_profiles`’s return value is
shaped to support without the agent eyeballing a table.

## Extending this to your own code

The same five steps apply to any script:

1.  Nothing to instrument up front if you’re happy with automatic
    whole-call-stack regions (`run_profile`’s default,
    `only_user_code=True` keeps it to your code only). Add
    `@ProfileManager.profile(...)` /
    `with ProfileManager.profile_region(...)` where you want named,
    consistently-boundaried regions instead — see
    `examples/ex_lazy_setup.py`.
2.  Pin a correctness check the same way `test_correctness.py` does: a
    reference value or golden output computed once, asserted every time.
3.  Run the loop above. `threshold_pct` on `compare_profiles` (default
    5%) controls what counts as a per-region regression/improvement
    worth surfacing — raise it on noisy machines, lower it once you have
    repeated runs to trust small deltas.

## Known gap: noise

This example runs once per side, which is fine for a change this large
(44% faster) but is not a rigorous benchmark — a single
`perf_counter_ns` run on a shared machine has enough jitter to
manufacture a “regression” on a change that didn’t actually regress
anything, particularly for sub-millisecond regions like `norm` above.
Repeating `run_profile` a few times per side and comparing
distributions, not points, is the natural next step and is not built yet
(see *Not yet built* in `CLAUDE.md`).
