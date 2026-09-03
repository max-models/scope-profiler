"""Tests for ``scope-profiler inspect``."""

import json

import h5py
import numpy as np
import pytest

from scope_profiler import read_h5
from scope_profiler.__main__ import main as cli_main
from scope_profiler.h5writer import ProfilingWriter
from scope_profiler.inspection import (
    collect_file_metadata,
    inspect_file,
)
from scope_profiler.inspection import main as inspect_main
from scope_profiler.inspection import (
    write_metadata_json,
)
from scope_profiler.likwid_data import LikwidRegionResult, write_likwid_results
from scope_profiler.perf_events import PerfEventTotals
from scope_profiler.profile_manager import RankPayload

NS = 1_000_000_000


def _write_sample_h5(path, rank_regions, metadata=None, sources=None):
    """``sources``: region name -> ``(source_file, source_lineno, source_text)``,
    written onto that region's group on every rank that has it."""
    sources = sources or {}
    with h5py.File(path, "w") as h5file:
        if metadata:
            meta_grp = h5file.create_group("metadata")
            for key, value in metadata.items():
                if isinstance(value, (list, tuple)):
                    meta_grp.attrs.create(key, list(value), dtype=h5py.string_dtype())
                else:
                    meta_grp.attrs[key] = value
        for rank, regions in rank_regions.items():
            regions_group = h5file.create_group(f"rank{rank}").create_group("regions")
            for region_name, payload in regions.items():
                region_group = regions_group.create_group(region_name)
                starts, ends = payload
                region_group.create_dataset(
                    "start_times",
                    data=np.asarray(starts, dtype=np.int64),
                )
                region_group.create_dataset(
                    "end_times",
                    data=np.asarray(ends, dtype=np.int64),
                )
                source = sources.get(region_name)
                if source is not None:
                    source_file, source_lineno, source_text = source
                    region_group.attrs["source_file"] = source_file
                    region_group.attrs["source_lineno"] = source_lineno
                    region_group.attrs["source_text"] = source_text


@pytest.fixture
def sample_file(tmp_path):
    """Two ranks, a cheap and an expensive region, with rich metadata."""
    path = tmp_path / "profiling_data.h5"
    _write_sample_h5(
        path,
        {
            0: {
                "setup": ([0], [1 * NS]),
                "solve": ([2 * NS, 5 * NS], [4 * NS, 8 * NS]),
            },
            1: {
                "setup": ([0], [3 * NS]),
                "solve": ([2 * NS, 5 * NS], [6 * NS, 9 * NS]),
            },
        },
        metadata={
            "timestamp": "2026-07-26T10:00:00",
            "user": "max",
            "hostname": "lrdn1234",
            "platform": "Linux-5.14.0-x86_64",
            "uname": "Linux lrdn1234 5.14.0 #1 SMP x86_64",
            "chip_information": "AMD EPYC 9654 96-Core Processor",
            "mpi_size": 2,
            "omp_num_threads": 8,
            "total_cores": 16,
            "modules": ["profile/base", "gcc/12.3.0", "python/3.11.7"],
            "SLURM_JOB_ID": "1234567",
            "SLURMD_NODENAME": "lrdn1234",
            "PATH": "/very/long/path" * 40,
            "VIRTUAL_ENV": "/home/max/.venv",
        },
    )
    return path


def test_inspect_prints_metadata_and_regions(sample_file, capsys):
    inspect_file(sample_file)
    out = capsys.readouterr().out

    # Header
    assert "profiling_data.h5" in out
    assert "2 rank(s), 2 region(s)" in out
    assert "wall clock" in out

    # Metadata, grouped
    assert "Metadata" in out
    assert "chip_information" in out and "AMD EPYC 9654" in out
    assert "uname" in out
    assert "SLURM_JOB_ID" in out and "1234567" in out
    assert "VIRTUAL_ENV" in out

    # Modules are listed one per line, not as a Python repr
    assert "Modules (3)" in out
    assert "    gcc/12.3.0" in out
    assert "['profile/base'" not in out

    # Region table with overall stats
    assert "Regions (2)" in out
    header = next(line for line in out.splitlines() if "total [s]" in line)
    assert "total [s]" in header and "avg [s]" in header
    assert "min [s]" not in header and "std [s]" not in header
    assert "setup" in out and "solve" in out
    assert "TOTAL" not in out


def test_inspect_prints_perf_event_totals(tmp_path, capsys):
    path = tmp_path / "perf_events.h5"
    payload = RankPayload(
        regions={"solve": (np.asarray([0]), np.asarray([NS]))},
        likwid={},
        likwid_environment={},
        perf_events={
            "solve": PerfEventTotals(
                calls=1,
                values={"cycles": 1234, "instructions": 5678},
            ),
        },
    )
    with ProfilingWriter(path) as writer:
        writer.write_rank(0, payload)

    inspect_file(path)
    out = capsys.readouterr().out

    assert "Perf events (rank 0)" in out
    assert "cycles" in out and "1234" in out
    assert "instructions" in out and "5678" in out


def test_region_statistics_are_seconds(sample_file, capsys):
    """solve: 2 s + 3 s on rank 0, 4 s + 4 s on rank 1."""
    inspect_file(
        sample_file,
        include="solve",
        columns=["region", "ranks", "calls", "total", "avg", "min", "max"],
    )
    line = next(
        line for line in capsys.readouterr().out.splitlines() if "solve" in line
    )
    fields = [field.strip() for field in line.strip("│ ").split("│")]

    assert fields[0] == "solve"
    assert fields[1] == "2"  # ranks
    assert fields[2] == "4"  # calls
    assert float(fields[3]) == pytest.approx(13.0)  # total
    assert float(fields[4]) == pytest.approx(3.2)  # avg, displayed to 1 decimal
    assert float(fields[5]) == pytest.approx(2.0)  # min
    assert float(fields[6]) == pytest.approx(4.0)  # max


def test_long_metadata_values_are_clipped_unless_full(sample_file, capsys):
    inspect_file(sample_file)
    clipped = next(
        line for line in capsys.readouterr().out.splitlines() if "PATH" in line
    )
    assert "[…]" in clipped

    inspect_file(sample_file, full=True)
    full = next(line for line in capsys.readouterr().out.splitlines() if "PATH" in line)
    assert "[…]" not in full
    assert len(full) > len(clipped)


def test_sorting(sample_file, capsys):
    inspect_file(sample_file, sort="total")
    ordered = [
        line.strip("│ ").split("│")[0].strip()
        for line in capsys.readouterr().out.splitlines()
        if "│ setup" in line or "│ solve" in line
    ]
    assert ordered == ["solve", "setup"]  # solve: 13 s, setup: 4 s

    inspect_file(sample_file, sort="name")
    ordered = [
        line.strip("│ ").split("│")[0].strip()
        for line in capsys.readouterr().out.splitlines()
        if "│ setup" in line or "│ solve" in line
    ]
    assert ordered == ["setup", "solve"]


def test_include_exclude_and_ranks(sample_file, capsys):
    inspect_file(sample_file, include="solve")
    out = capsys.readouterr().out
    assert "solve" in out and "Regions (1)" in out

    inspect_file(sample_file, exclude="solve")
    out = capsys.readouterr().out
    assert "Regions (1)" in out and "setup" in out

    # Restricting to rank 0 halves solve's call count (2 of 4).
    inspect_file(sample_file, include="solve", ranks=[0])
    line = next(
        line for line in capsys.readouterr().out.splitlines() if "solve" in line
    )
    fields = [field.strip() for field in line.strip("│ ").split("│")]
    assert fields[1] == "2"  # calls; ranks is no longer in the default table


def test_section_switches(sample_file, capsys):
    inspect_file(sample_file, show_regions=False)
    out = capsys.readouterr().out
    assert "Metadata" in out and "Regions" not in out

    inspect_file(sample_file, show_metadata=False)
    out = capsys.readouterr().out
    assert "Metadata" not in out and "Regions" in out


def test_source_prints_captured_call_site(tmp_path, capsys):
    path = tmp_path / "with_source.h5"
    _write_sample_h5(
        path,
        {0: {"solve": ([0], [1 * NS])}},
        sources={
            "solve": (
                "kernels.py",
                12,
                '    with ProfileManager.profile_region("solve"):\n        pass\n',
            ),
        },
    )

    inspect_file(path, source=["solve"])
    out = capsys.readouterr().out

    assert "Source (1)" in out
    assert "solve  (kernels.py:12)" in out
    assert 'with ProfileManager.profile_region("solve")' in out


def test_source_without_a_captured_region_says_so(sample_file, capsys):
    inspect_file(sample_file, source=["solve"])
    out = capsys.readouterr().out

    assert "solve: source not captured" in out


def test_source_of_an_unknown_region_lists_the_available_ones(sample_file, capsys):
    inspect_file(sample_file, source=["nope"])
    out = capsys.readouterr().out

    assert "'nope': no such region" in out
    assert "'setup'" in out and "'solve'" in out


def test_source_prints_regardless_of_metadata_only(sample_file, capsys):
    """--source is independent of --metadata-only/--regions-only."""
    inspect_file(sample_file, show_regions=False, source=["solve"])
    out = capsys.readouterr().out

    assert "Regions" not in out
    assert "Source (1)" in out


def test_cli_source_flag(tmp_path, capsys):
    path = tmp_path / "with_source.h5"
    _write_sample_h5(
        path,
        {0: {"solve": ([0], [1 * NS])}},
        sources={"solve": ("kernels.py", 3, "    with region():\n        pass\n")},
    )

    inspect_main([str(path), "--source", "solve"])
    out = capsys.readouterr().out

    assert "Source (1)" in out
    assert "kernels.py:3" in out


def test_file_without_regions_or_metadata(tmp_path, capsys):
    path = tmp_path / "empty.h5"
    _write_sample_h5(path, {})

    inspect_file(path)
    out = capsys.readouterr().out

    assert "0 rank(s), 0 region(s)" in out
    assert "(none recorded)" in out


def test_cli_entry_point(sample_file, capsys):
    inspect_main([str(sample_file)])
    assert "Metadata" in capsys.readouterr().out


def test_cli_accepts_region_table_columns(sample_file, capsys):
    inspect_main(
        [
            str(sample_file),
            "--regions-only",
            "--columns",
            "region",
            "ranks",
            "calls",
            "total",
            "avg",
        ],
    )
    out = capsys.readouterr().out
    header = next(line for line in out.splitlines() if "total [s]" in line)

    assert "total [s]" in header and "avg [s]" in header
    assert "min [s]" not in header
    assert "imbalance [%]" not in header
    assert "setup" in out and "solve" in out and "TOTAL" not in out


def test_cli_accepts_multiple_files_and_globs(sample_file, capsys):
    second = sample_file.parent / "second_run.h5"
    _write_sample_h5(second, {0: {"setup": ([0], [1 * NS])}}, metadata={"user": "max"})

    inspect_main([str(sample_file), str(second)])
    assert capsys.readouterr().out.count("Metadata") == 2

    # A glob covers both, and repeated paths are reported once (as in plot).
    inspect_main([str(sample_file.parent / "*.h5"), str(sample_file)])
    assert capsys.readouterr().out.count("Metadata") == 2


def test_dispatch_from_main_cli(sample_file, capsys):
    cli_main(["inspect", str(sample_file), "--metadata-only"])
    out = capsys.readouterr().out

    assert "Metadata" in out
    assert "Regions" not in out


def test_write_metadata_json(sample_file, tmp_path):
    out_file = tmp_path / "exported" / "metadata.json"
    returned = write_metadata_json(sample_file, out_file)

    # Parent directories are created as needed.
    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload == returned

    entry = payload["files"][0]
    assert entry["file_path"] == str(sample_file.resolve())
    assert entry["num_ranks"] == 2

    metadata = entry["metadata"]
    assert metadata["chip_information"] == "AMD EPYC 9654 96-Core Processor"
    assert metadata["SLURM_JOB_ID"] == "1234567"
    assert metadata["modules"] == ["profile/base", "gcc/12.3.0", "python/3.11.7"]
    # numpy scalars from HDF5 must survive as plain JSON numbers.
    assert metadata["mpi_size"] == 2
    assert isinstance(metadata["mpi_size"], int)
    # Export is never clipped, unlike the printed output.
    assert "[…]" not in metadata["PATH"]
    assert len(metadata["PATH"]) == len("/very/long/path" * 40)


def test_write_metadata_json_accepts_readers_and_sequences(sample_file, tmp_path):
    second = tmp_path / "second_run.h5"
    _write_sample_h5(second, {0: {"setup": ([0], [1 * NS])}}, metadata={"user": "max"})

    payload = write_metadata_json(
        [read_h5(sample_file), second],
        tmp_path / "both.json",
    )

    assert len(payload["files"]) == 2
    assert payload["files"][1]["metadata"]["user"] == "max"


def test_collect_file_metadata_without_writing(sample_file):
    payload = collect_file_metadata(sample_file)

    assert list(payload) == ["files"]
    assert payload["files"][0]["metadata"]["hostname"] == "lrdn1234"


def test_cli_export_metadata(sample_file, tmp_path, capsys):
    out_file = tmp_path / "metadata.json"
    inspect_main([str(sample_file), "--export-metadata", str(out_file)])

    out = capsys.readouterr().out
    assert "Metadata" in out  # summary is still printed
    assert f"Metadata written to {out_file}" in out

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["files"][0]["metadata"]["SLURM_JOB_ID"] == "1234567"


def test_cli_export_metadata_quiet(sample_file, tmp_path, capsys):
    out_file = tmp_path / "metadata.json"
    inspect_main([str(sample_file), "--export-metadata", str(out_file), "--quiet"])

    out = capsys.readouterr().out
    assert "Regions" not in out
    assert "chip_information" not in out
    assert out.strip() == f"Metadata written to {out_file}"
    assert out_file.exists()


def test_cli_export_metadata_multiple_files(sample_file, tmp_path, capsys):
    second = tmp_path / "second_run.h5"
    _write_sample_h5(second, {0: {"setup": ([0], [1 * NS])}}, metadata={"user": "max"})
    out_file = tmp_path / "metadata.json"

    inspect_main(
        [str(sample_file), str(second), "--export-metadata", str(out_file), "--quiet"],
    )

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert [entry["file_path"] for entry in payload["files"]] == [
        str(sample_file.resolve()),
        str(second.resolve()),
    ]


def test_inspect_listed_in_top_level_help(capsys):
    with pytest.raises(SystemExit):
        cli_main(["--help"])

    assert "inspect" in capsys.readouterr().out


def _likwid_result(tag, group="CLOCK"):
    """A minimal LIKWID result for one region on one rank."""
    return LikwidRegionResult(
        tag=tag,
        group_id=0,
        group_name=group,
        cpus=[0],
        times=np.full(1, 0.5),
        call_counts=np.full(1, 3, dtype=np.int64),
        event_names=["INSTR_RETIRED_ANY", "CAS_COUNT_RD"],
        counter_names=["FIXC0", "MBOX0C0"],
        events=np.array([[100.0], [10.0]]),
        metric_names=["Clock [MHz]", "CPI"],
        metrics=np.array([[2400.0], [0.5]]),
        source="full_api",
    )


def _write_likwid_h5(path, tag="solve"):
    """A merged-looking file with one timed region and its LIKWID counters."""
    with h5py.File(path, "w") as h5file:
        grp = h5file.create_group("rank0")
        region_group = grp.create_group("regions").create_group(tag)
        region_group.create_dataset("start_times", data=np.array([0], dtype=np.int64))
        region_group.create_dataset("end_times", data=np.array([10**9], dtype=np.int64))
        write_likwid_results(grp, [_likwid_result(tag)])
    return path


def test_inspect_prints_likwid_tables_when_present(tmp_path, capsys):
    """inspect prints the LIKWID counter table alongside the region table."""
    path = _write_likwid_h5(tmp_path / "d.h5")

    inspect_file(path)
    out = capsys.readouterr().out

    assert "solve" in out
    assert "LIKWID counters (rank 0, group CLOCK)" in out
    assert "CAS_COUNT_RD" in out
    assert "Clock [MHz]" in out


def test_inspect_without_likwid_prints_only_regions(sample_file, capsys):
    """Files recorded without LIKWID data print no counter section."""
    inspect_file(sample_file)
    out = capsys.readouterr().out

    assert "solve" in out
    assert "LIKWID" not in out
