"""Combining result sets, as a mixed-language run has to."""

import numpy as np
import pytest

from scope_profiler import ProfilingResults
from scope_profiler.mpi_region import MPIRegion
from scope_profiler.region import Region
from scope_profiler.results import merge_results

NS = 1_000_000_000


def results(regions: dict, metadata: dict | None = None, **kwargs) -> ProfilingResults:
    """A result set from ``{name: {rank: (starts, ends)}}``."""
    return ProfilingResults(
        {
            name: MPIRegion(
                name=name,
                regions={
                    rank: Region(
                        np.asarray(starts, dtype=np.int64),
                        np.asarray(ends, dtype=np.int64),
                    )
                    for rank, (starts, ends) in ranks.items()
                },
            )
            for name, ranks in regions.items()
        },
        metadata=metadata or {},
        **kwargs,
    )


def test_regions_from_both_sets_end_up_in_one():
    python = results({"python:solve": {0: ([0], [4 * NS])}}, {"label": "run a"})
    fortran = results({"fortran:kernel": {0: ([NS], [3 * NS])}}, {"source": "fortran"})

    merged = merge_results(python, fortran)

    assert sorted(merged.region_names) == ["fortran:kernel", "python:solve"]
    assert merged["python:solve"].total_duration == pytest.approx(4.0)
    assert merged["fortran:kernel"].total_duration == pytest.approx(2.0)
    # The driver's metadata describes the run; the other set fills gaps.
    assert merged.label == "run a"
    assert merged.metadata["source"] == "fortran"


def test_ranks_line_up_across_sets():
    python = results({"py": {0: ([0], [NS]), 1: ([0], [2 * NS])}})
    fortran = results({"fo": {0: ([0], [3 * NS]), 1: ([0], [4 * NS])}})

    merged = merge_results(python, fortran)

    assert merged.num_ranks == 2
    assert list(merged["fo"].regions) == [0, 1]
    assert merged["fo"].regions[1].total_duration == pytest.approx(4.0)


def test_a_shared_region_name_is_refused():
    """Silently merging them would double-count a wrapper and its callee."""
    python = results({"solve": {0: ([0], [4 * NS])}})
    fortran = results({"solve": {0: ([NS], [3 * NS])}})

    with pytest.raises(ValueError, match="appears in more than one result set"):
        merge_results(python, fortran)


def test_the_same_name_within_one_set_is_not_a_collision():
    """A region spanning ranks is normal; only cross-set duplicates are not."""
    python = results({"solve": {0: ([0], [NS]), 1: ([0], [2 * NS])}})

    merged = merge_results(python)

    assert merged["solve"].num_calls == 2


def test_non_root_sets_are_ignored():
    """So a parallel script can merge unguarded, like everything else."""
    root = results({"py": {0: ([0], [NS])}})
    other_rank = ProfilingResults({}, num_ranks=4, is_root=False)

    merged = merge_results(root, other_rank)

    assert merged.region_names == ["py"]
    assert merged.is_root


def test_merging_only_non_root_sets_stays_quiet():
    empty = ProfilingResults({}, num_ranks=4, is_root=False)

    merged = merge_results(empty, empty)

    assert not merged.is_root
    assert merged.region_names == []


def test_label_and_file_path_can_be_set_on_the_merge():
    merged = merge_results(
        results({"py": {0: ([0], [NS])}}),
        results({"fo": {0: ([0], [NS])}}),
        label="mixed run",
        file_path="combined.h5",
    )

    assert merged.label == "mixed run"
    assert merged.file_path.name == "combined.h5"


def test_merging_nothing_is_an_error():
    with pytest.raises(ValueError, match="at least one result set"):
        merge_results()


def test_likwid_counters_survive_the_merge():
    from scope_profiler.likwid_data import LikwidRegionResult

    counters = {0: {"solve": LikwidRegionResult(tag="solve", group_name="FLOPS_DP")}}
    python = ProfilingResults({}, likwid=counters)
    fortran = results({"fo": {0: ([0], [NS])}})

    merged = merge_results(python, fortran)

    assert merged.has_likwid
    assert merged.get_likwid_region("solve", rank=0).group_name == "FLOPS_DP"
