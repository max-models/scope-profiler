"""Capturing and round-tripping a region's call-site source (issue #161)."""

import pytest

from scope_profiler import ProfileManager, read_h5


@pytest.fixture(autouse=True)
def _reset():
    yield
    ProfileManager._reset()


def test_location_is_captured_but_source_text_is_off_by_default(tmp_path):
    """The cheap location default avoids reading/parsing the source file."""
    path = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(path))

    with ProfileManager.profile_region("solve"):
        pass

    results = ProfileManager.finalize(verbose=False, return_results=True)
    region = results["solve"]

    assert region.has_source
    assert region.source_file == __file__
    assert region.source_text is None


def test_context_manager_region_captures_its_with_block(tmp_path):
    """A ``with`` region records the file, line and text of its own block."""
    path = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(path), capture_region_source=True)

    with ProfileManager.profile_region("solve"):
        pass

    results = ProfileManager.finalize(verbose=False, return_results=True)
    region = results["solve"]

    assert region.has_source
    assert region.source_file == __file__
    assert "profile_region" in region.source_text
    assert "solve" in region.source_text


def test_decorated_region_captures_the_function_source(tmp_path):
    """A decorated region records the whole decorated function, not one line."""
    path = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(path), capture_region_source=True)

    @ProfileManager.profile("kernel")
    def kernel():
        pass

    kernel()

    results = ProfileManager.finalize(verbose=False, return_results=True)
    region = results["kernel"]

    assert region.has_source
    assert "def kernel" in region.source_text


def test_reusing_a_region_name_keeps_the_first_call_site(tmp_path):
    """Same-named regions from different call sites are combined (see the
    region's own docs); the source keeps whichever call site created it
    first, rather than silently overwriting or erroring.
    """
    path = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(path), capture_region_source=True)

    def first():
        with ProfileManager.profile_region("shared"):
            pass

    def second():
        with ProfileManager.profile_region("shared"):
            pass

    first()
    second()

    results = ProfileManager.finalize(verbose=False, return_results=True)
    region = results["shared"]

    assert region.has_source
    assert "def first" not in region.source_text  # only the with-block, not the def
    # The captured line is inside first(), not second().
    import inspect

    first_lineno = inspect.getsourcelines(first)[1]
    second_lineno = inspect.getsourcelines(second)[1]
    assert first_lineno < region.source_lineno < second_lineno


def test_source_is_not_captured_for_disabled_profiling(tmp_path):
    """Disabled regions do no capturing (and no-op cheaply) either."""
    path = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(path), deactivate_profiling=True)

    with ProfileManager.profile_region("off"):
        pass

    region = ProfileManager.get_region("off")
    assert not region.source_text


def test_source_round_trips_through_the_written_file(tmp_path):
    """The source captured in-process survives a write and re-read."""
    path = tmp_path / "profiling_data.h5"
    ProfileManager.setup(file_path=str(path), capture_region_source=True)

    with ProfileManager.profile_region("written"):
        pass

    ProfileManager.finalize(verbose=False)
    from_disk = read_h5(str(path))
    region = from_disk["written"]

    assert region.has_source
    assert region.source_file == __file__
    assert "profile_region" in region.source_text


def test_files_without_source_attrs_read_back_as_none(tmp_path):
    """Older files written before this feature still read back cleanly."""
    import numpy as np

    from scope_profiler.h5writer import ProfilingWriter
    from scope_profiler.profile_manager import RankPayload

    path = tmp_path / "legacy.h5"
    with ProfilingWriter(path) as writer:
        writer.write_rank(
            0,
            RankPayload(
                regions={"solve": (np.array([0]), np.array([1]))},
                likwid={},
                likwid_environment={},
            ),
        )

    region = read_h5(path)["solve"]
    assert not region.has_source
    assert region.source_file is None
    assert region.source_lineno is None
    assert region.source_text is None
