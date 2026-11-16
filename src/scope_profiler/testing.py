"""Test helpers for libraries that ship opt-in instrumentation.

A library that registers an opt-in tag (see :func:`scope_profiler.library`)
checks it with one test::

    from scope_profiler.testing import assert_opt_in_tag

    def test_the_profiling_regions_are_opt_in():
        assert_opt_in_tag("mylib", run=solve_a_small_case, expected={"mylib.solve"})

and uses the ``fresh_profiler`` fixture (from the pytest plugin, no import
needed) in tests of its own that profile.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from importlib.metadata import entry_points

from scope_profiler.profile_manager import ProfileManager

ENTRY_POINT_GROUP = "scope_profiler.opt_in_tags"
_ENV_VAR = "SCOPE_PROFILER_ENABLE_TAGS"
#: The application region around ``run`` in :func:`recorded_region_names`.
RUN_REGION = "scope_profiler.testing.run"


@contextmanager
def fresh_profiler() -> Iterator[None]:
    """Run a block with a fresh profiler that no environment variable enables.

    Resets the profiling state on entry and exit and hides
    ``SCOPE_PROFILER_ENABLE_TAGS``, so the block sees what an application
    that does not ask for any tag sees. The ``fresh_profiler`` pytest fixture
    is this context manager.

    Yields
    ------
    None
    """
    saved = os.environ.pop(_ENV_VAR, None)
    ProfileManager.reset()
    try:
        yield
    finally:
        ProfileManager.reset()
        if saved is not None:
            os.environ[_ENV_VAR] = saved


def recorded_region_names(run: Callable[[], object], **setup_kwargs) -> set[str]:
    """Return the names of the regions a call of ``run`` records.

    Profiles ``run()`` inside the application region :data:`RUN_REGION`, with
    ``setup(output=None, **setup_kwargs)``, in a :func:`fresh_profiler`.

    Parameters
    ----------
    run : callable
        Called once without arguments.
    **setup_kwargs
        Passed to :meth:`ProfileManager.setup`, e.g. ``enable_tags=["mylib"]``.

    Returns
    -------
    set of str
        The recorded region names, :data:`RUN_REGION` included. Under MPI the
        run is gathered on rank 0; the other ranks get an empty set.
    """
    with fresh_profiler():
        ProfileManager.setup(output=None, **setup_kwargs)
        with ProfileManager.region(RUN_REGION):
            run()
        results = ProfileManager.finalize(verbose=False, return_results=True)
        return set(results.region_names)


def assert_opt_in_tag(
    tag: str,
    run: Callable[[], object],
    expected: Iterable[str],
    *,
    module: str | None = None,
) -> None:
    """Check that ``tag`` is a working opt-in tag of an installed package.

    Asserts that

    - ``run()`` records none of the tag's regions (named ``"<tag>.*"``) by
      default;
    - it records at least the ``expected`` regions with
      ``enable_tags=[tag]`` and with ``enable_tags=["*"]``;
    - the package declares ``tag`` as a ``scope_profiler.opt_in_tags`` entry
      point, so that ``scope-profiler tags`` finds it.

    Under MPI every rank must call it; the region checks run on rank 0.

    Parameters
    ----------
    tag : str
        The opt-in tag.
    run : callable
        Called without arguments; exercises some of the tagged regions.
    expected : iterable of str
        Full names of regions that ``run`` must record once the tag is on.
    module : str, optional
        The module the entry point must name; defaults to ``tag``.

    Raises
    ------
    AssertionError
        If any of the checks fails.
    """
    expected = set(expected)
    prefix = f"{tag}."
    stray = {name for name in expected if not name.startswith(prefix)}
    assert not stray, f"expected regions not named {prefix!r}*: {sorted(stray)}"

    off = recorded_region_names(run)
    leaked = {name for name in off if name.startswith(prefix)}
    assert not leaked, (
        f"regions of {tag!r} recorded without enabling it: {sorted(leaked)}"
    )

    for enable in ([tag], ["*"]):
        names = recorded_region_names(run, enable_tags=enable)
        if not names:  # not rank 0: the run was gathered there
            continue
        missing = expected - names
        assert not missing, (
            f"with enable_tags={enable} the regions {sorted(missing)} were not "
            f"recorded; recorded: {sorted(names)}"
        )

    declared = {ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}
    want = tag if module is None else module
    assert declared.get(tag) == want, (
        f"no entry point {tag} = {want!r} in group {ENTRY_POINT_GROUP!r} "
        f"(declared: {declared}); add it to pyproject.toml under "
        f'[project.entry-points."{ENTRY_POINT_GROUP}"]'
    )
