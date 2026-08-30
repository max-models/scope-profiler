"""Manually profile every tenth timestep.

Run with::

    python examples/ex_pause_resume.py

This shows the explicit pause/resume form. The automatic ``sample_every``
helper is useful for simple loops, but manual control is convenient when the
decision depends on additional simulation state.
"""

from scope_profiler import ProfileManager


def advance_simulation(timestep: int) -> int:
    """Stand-in for one simulation timestep."""
    return sum((value + timestep) % 17 for value in range(10_000))


def main(num_steps: int = 25) -> None:
    with ProfileManager.session(
        file_path="pause_resume_profile.h5",
        verbose=False,
        return_results=True,
    ) as run:
        step = ProfileManager.profile_region("simulation.step")
        for timestep in range(num_steps):
            if timestep % 10 == 0:
                ProfileManager.resume()
            else:
                ProfileManager.pause()

            with step:
                advance_simulation(timestep)

    results = run.results
    assert results["simulation.step"][0].num_calls == 3
    results.print_summary(title="Manual pause/resume profiling")


if __name__ == "__main__":
    main()
