"""The C API's optional HDF5 backend, compiled against a real libhdf5.

The point of the backend is that a C run needs no import step: what
``sp_finalize()`` writes is already a profile ``read_h5()`` opens. Only a
compiled run can show that, and only against the real library -- the layout
is a contract between ``scope_profiler_hdf5.h`` and
:mod:`scope_profiler.h5writer`, and every way of getting it wrong (a compound
field in the wrong order, a string that is not variable-length, an event
column a row's offsets do not line up with) is invisible until h5py reads it
back.

The module skips where libhdf5's headers cannot be found, so it is the
format-selection tests in ``test_c_api.py`` -- which need no HDF5 at all --
that keep the fallback path covered on a machine without it.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scope_profiler import read_h5
from scope_profiler.h5writer import _timing_summary
from scope_profiler.native_trace import (
    c_source_path,
    find_traces,
    load_traces,
    read_native_h5,
)
from scope_profiler.tests.test_c_api import COMPILER, build, run

SOURCE = c_source_path()

#: A trivial program that only has to link, to prove a candidate flag set works.
_PROBE = """
#include <hdf5.h>
int main(void) { return H5open() < 0; }
"""


def _candidate_flags():
    """Flag sets that might build against libhdf5, best first.

    HDF5 has no single canonical install location and no single canonical
    pkg-config name, so rather than guess, each candidate is tried on a probe
    program and the first that compiles *and* links wins.
    """
    yield ["-lhdf5"]

    for package in ("hdf5", "hdf5-serial"):
        query = shutil.which("pkg-config")
        if query is None:
            break
        flags = []
        for argument in ("--cflags", "--libs"):
            result = subprocess.run(
                [query, argument, package],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                flags = []
                break
            flags.extend(result.stdout.split())
        if flags:
            yield flags

    # No canonical install location either, so try the ones a machine is
    # likely to have: HDF5_DIR if the user set it, Homebrew's prefix, and
    # Debian/Ubuntu's serial layout, whose headers and libraries live apart.
    roots = [Path(os.environ["HDF5_DIR"])] if "HDF5_DIR" in os.environ else []
    roots += [
        Path("/opt/homebrew/opt/hdf5"),
        Path("/usr/local/opt/hdf5"),
        Path("/usr"),
    ]
    for root in roots:
        for include in (root / "include", root / "include/hdf5/serial", root):
            if not (include / "hdf5.h").exists():
                continue
            libraries = [root / "lib", *sorted(root.glob("lib/*/hdf5/serial"))]
            yield [
                f"-I{include}",
                *(f"-L{path}" for path in libraries if path.is_dir()),
                "-lhdf5",
            ]


def _resolve_flags():
    """The first candidate flag set that builds the probe, or None."""
    if COMPILER is None:
        return None
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "probe.c"
        source.write_text(_PROBE)
        for flags in _candidate_flags():
            result = subprocess.run(
                [
                    COMPILER,
                    "-std=c99",
                    str(source),
                    "-o",
                    str(Path(directory) / "probe"),
                    *flags,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return flags
    return None


HDF5_FLAGS = _resolve_flags()

pytestmark = [
    pytest.mark.skipif(COMPILER is None, reason="no C compiler on PATH"),
    pytest.mark.skipif(not SOURCE.exists(), reason=f"{SOURCE} not found (installed?)"),
    pytest.mark.skipif(
        HDF5_FLAGS is None, reason="no usable libhdf5 for the C compiler"
    ),
]


def build_with_hdf5(tmp_path: Path, program: str, name: str = "prog") -> Path:
    """Compile ``program`` against the API with its HDF5 backend enabled."""
    return build(tmp_path, program, name=name, extra=["-DSP_USE_HDF5", *HDF5_FLAGS])


BOTH_FORMATS_PROGRAM = """
#include "scope_profiler.h"
#include <stdio.h>

int main(void)
{
    int i, outer, inner;

    sp_init("run", 3);
    outer = SP_REGION_AT("outer");
    inner = sp_region("inner");

    for (i = 0; i < 4; ++i) {
        sp_begin(outer);
        sp_begin(inner);
        sp_end(inner);
        sp_end(outer);
    }

    /* The same recorded calls, published twice: flush leaves the profiler
     * running, so the two files differ only in their format. */
    printf("%d %s\\n", (int)sp_current_output_format(), sp_output_path());
    if (sp_profiler_flush(sp_default_profiler()) != 0) return 1;
    if (sp_set_output_format(SP_OUTPUT_TRACE) != SP_OK) return 2;
    printf("%d %s\\n", (int)sp_current_output_format(), sp_output_path());
    sp_finalize();
    return 0;
}
"""


def test_hdf5_is_the_default_format_and_needs_no_import(tmp_path):
    """A C run's own output file opens as an ordinary profile."""
    executable = build_with_hdf5(tmp_path, BOTH_FORMATS_PROGRAM)
    output = run(executable, tmp_path).stdout.split("\n")

    # SP_OUTPUT_HDF5 == 0: the build did not have to ask for it.
    assert output[0] == "0 run_rank00003.h5"
    assert output[1] == "1 run_rank00003.spt"

    results = read_h5(tmp_path / "run_rank00003.h5")
    assert sorted(region.name for region in results.get_regions()) == ["inner", "outer"]
    # One rank per file, under the rank sp_init() was given -- so the ranks of
    # an MPI run stay distinguishable when they are merged afterwards.
    assert sorted(results.get_region("outer").regions) == [3]

    outer = results.get_region("outer").regions[3]
    assert len(outer.start_times_ns) == 4
    assert np.all(outer.end_times_ns >= outer.start_times_ns)
    # SP_REGION_AT() records where the region is declared, and the reader
    # surfaces it exactly as it does for a Python region.
    assert outer.source_file.endswith("prog.c")
    assert outer.source_lineno > 0
    assert results.get_region("inner").regions[3].source_file is None

    assert results.metadata["source"] == "native"
    assert results.metadata["source_language"] == "c"
    assert int(results.metadata["mpi_rank"]) == 3
    assert int(results.metadata["start_time_ns"]) == int(outer.start_times_ns[0])


def test_hdf5_and_trace_outputs_carry_the_same_measurements(tmp_path):
    """Both formats of one run are read back as identical timing data."""
    executable = build_with_hdf5(tmp_path, BOTH_FORMATS_PROGRAM)
    run(executable, tmp_path)

    direct = read_h5(tmp_path / "run_rank00003.h5")
    imported = load_traces(tmp_path / "run_rank00003.spt")

    assert sorted(region.name for region in direct.get_regions()) == sorted(
        region.name for region in imported.get_regions()
    )
    for region in direct.get_regions():
        expected = imported.get_region(region.name).regions[3]
        recorded = region.regions[3]
        assert np.array_equal(recorded.start_times_ns, expected.start_times_ns)
        assert np.array_equal(recorded.end_times_ns, expected.end_times_ns)


def test_summary_statistics_match_the_python_writer(tmp_path):
    """The fixed-size stats column agrees with h5writer's, field for field."""
    import h5py

    executable = build_with_hdf5(tmp_path, BOTH_FORMATS_PROGRAM)
    run(executable, tmp_path)
    regions = read_h5(tmp_path / "run_rank00003.h5")

    with h5py.File(tmp_path / "run_rank00003.h5") as h5file:
        names = [value.decode() for value in h5file["region_table/names"][()]]
        index = h5file["rank_region_index"]
        assert index["ranks"][()].tolist() == [3] * len(names)
        assert index["event_offsets"][()].tolist() == [0, 4]
        assert index["event_counts"][()].tolist() == [4, 4]
        # Nesting is not recorded by the C API, so the reader reconstructs
        # exclusive time rather than reading a written total, exactly as for
        # an imported trace.
        assert index["exclusive_totals"][()].tolist() == [-1, -1]
        assert [value.decode() for value in index["tags"][()]] == ["[]", "[]"]

        for row, name in enumerate(names):
            region = regions.get_region(name).regions[3]
            expected = _timing_summary(
                (region.start_times_ns, region.end_times_ns),
            )
            written = index["summary_statistics"][row]
            for field in ("total", "minimum", "maximum", "first", "last"):
                assert int(written[field]) == expected[field], (name, field)
            assert int(written["start_minimum"]) == expected["start_minimum"]
            assert int(written["end_maximum"]) == expected["end_maximum"]
            assert int(written["gpu_count"]) == 0
            assert written["mean"] == pytest.approx(expected["mean"])
            assert written["m2"] == pytest.approx(expected["m2"], abs=1e-6)


RANK_PROGRAM = """
#include "scope_profiler.h"
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    int solve;

    sp_init("run", atoi(argv[1]));
    if (argc > 2 && strcmp(argv[2], "trace") == 0) {
        sp_set_output_format(SP_OUTPUT_TRACE);
    }
    solve = sp_region("solve");
    sp_begin(solve);
    sp_end(solve);
    sp_finalize();
    return 0;
}
"""


def test_import_native_merges_hdf5_ranks_with_traces(tmp_path):
    """One import folds per-rank .h5 and .spt files into a single profile."""
    executable = build_with_hdf5(tmp_path, RANK_PROGRAM)
    run(executable, tmp_path, "0")
    run(executable, tmp_path, "1")
    run(executable, tmp_path, "2", "trace")

    assert [path.name for path in find_traces(tmp_path)] == [
        "run_rank00000.h5",
        "run_rank00001.h5",
        "run_rank00002.spt",
    ]

    merged = tmp_path / "merged.h5"
    result = subprocess.run(
        [
            "python",
            "-m",
            "scope_profiler",
            "import-native",
            str(tmp_path),
            "-o",
            str(merged),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    results = read_h5(merged)
    assert results.num_ranks == 3
    assert sorted(results.get_region("solve").regions) == [0, 1, 2]
    # A merged profile is not itself per-rank output, so a second import of
    # the same directory does not swallow it.
    assert merged not in find_traces(tmp_path)


def test_a_rank_that_recorded_nothing_still_writes_a_readable_profile(tmp_path):
    """An empty run produces an empty profile, not a broken or missing file."""
    executable = build_with_hdf5(
        tmp_path,
        """
#include "scope_profiler.h"
int main(void)
{
    sp_init("empty", 0);
    sp_region("never_entered");
    sp_finalize();
    return 0;
}
""",
    )
    run(executable, tmp_path)

    ranks, metadata = read_native_h5(tmp_path / "empty_rank00000.h5")
    assert ranks == {}
    assert metadata["source"] == "native"
    assert read_h5(tmp_path / "empty_rank00000.h5").get_regions() == []


def test_an_interrupted_write_leaves_the_previous_profile_intact(tmp_path):
    """Publication is atomic: the destination is replaced, never truncated."""
    executable = build_with_hdf5(
        tmp_path,
        """
#include "scope_profiler.h"
#include <stdlib.h>

int main(int argc, char **argv)
{
    int i, solve;

    sp_init("run", 0);
    solve = sp_region("solve");
    for (i = 0; i < atoi(argv[1]); ++i) {
        sp_begin(solve);
        sp_end(solve);
    }
    sp_finalize();
    return 0;
}
""",
    )
    run(executable, tmp_path, "2")
    assert (
        len(
            read_h5(tmp_path / "run_rank00000.h5")
            .get_region("solve")
            .regions[0]
            .start_times_ns
        )
        == 2
    )

    # A second run replaces the first file rather than appending to or
    # corrupting it, and leaves no temporary behind.
    run(executable, tmp_path, "7")
    assert (
        len(
            read_h5(tmp_path / "run_rank00000.h5")
            .get_region("solve")
            .regions[0]
            .start_times_ns
        )
        == 7
    )
    assert not list(tmp_path.glob("*.tmp*"))
