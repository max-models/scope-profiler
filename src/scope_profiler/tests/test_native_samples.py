"""Line sampling of C regions, compiled and run for real.

The sampler lives in the C library, the address-to-line mapping in
``scope_profiler.native_samples``; only a real build with ``-g`` shows that the
two agree on which line was hot.
"""

import platform
import shutil

import pytest

from scope_profiler.__main__ import main
from scope_profiler.native_samples import (
    SampleFormatError,
    find_sample_files,
    line_profile,
    read_samples,
)
from scope_profiler.tests.test_c_api import COMPILER, SOURCE, build, run

pytestmark = [
    pytest.mark.skipif(COMPILER is None, reason="no C compiler on PATH"),
    pytest.mark.skipif(not SOURCE.exists(), reason=f"{SOURCE} not found (installed?)"),
    pytest.mark.skipif(
        platform.system() not in ("Linux", "Darwin"),
        reason="line sampling needs POSIX signals",
    ),
]

needs_symbolizer = pytest.mark.skipif(
    not (
        shutil.which("atos")
        or shutil.which("llvm-symbolizer")
        or shutil.which("addr2line")
    ),
    reason="no atos, llvm-symbolizer or addr2line",
)

#: Each region runs its own function until a deadline, so the sample count does
#: not depend on how fast the machine is, and the region's samples must land
#: in that function's lines.
PROGRAM = """
#include "scope_profiler.h"
#include <stdio.h>

static volatile double sink;

static void hot_loop(int64_t until)
{
    volatile double a = 0.0;
    volatile double b = 0.0;
    while (sp_now_ns() < until) {
        int i;
        for (i = 0; i < 2000; ++i) {
            a += i * 0.5;
            b += a * 1.0001 + b * 0.5 - a * b;
        }
        sink = a + b;
    }
}

static void cold_loop(int64_t until)
{
    volatile double a = 0.0;
    while (sp_now_ns() < until) {
        int i;
        for (i = 0; i < 2000; ++i) {
            a += i * 0.5;
        }
        sink = a;
    }
}

int main(void)
{
    int hot, cold;

    sp_init("prof", 0);
    hot = sp_region("hot");
    cold = sp_region("cold");
    if (sp_start_sampling(1000) != SP_OK) {
        return 3;
    }

    sp_begin(hot);
    hot_loop(sp_now_ns() + 500000000LL);
    sp_end(hot);

    sp_begin(cold);
    cold_loop(sp_now_ns() + 500000000LL);
    sp_end(cold);

    sp_finalize();
    return 0;
}
"""

#: The sampler must stay out of the way unless asked for.
PLAIN_PROGRAM = """
#include "scope_profiler.h"

int main(void)
{
    int id;

    sp_init("prof", 0);
    id = sp_region("work");
    sp_begin(id);
    sp_end(id);
    sp_finalize();
    return 0;
}
"""


def line_range(function: str) -> range:
    """Source lines of ``function`` in PROGRAM, from its signature to its brace."""
    lines = PROGRAM.splitlines()
    start = next(n for n, text in enumerate(lines, 1) if f"{function}(" in text)
    end = next(n for n, text in enumerate(lines, 1) if n > start and text == "}")
    return range(start, end + 1)


def test_no_samples_file_unless_sampling(tmp_path):
    run(build(tmp_path, PLAIN_PROGRAM, extra=("-g",)), tmp_path)

    assert not list(tmp_path.glob("*.sps"))


def test_sampling_through_the_environment(tmp_path, monkeypatch):
    executable = build(tmp_path, PLAIN_PROGRAM, extra=("-g",))
    monkeypatch.setenv("SCOPE_PROFILER_SAMPLE_HZ", "500")

    run(executable, tmp_path)

    # Started by sp_init() and written at finalize, even with no samples taken.
    (path,) = tmp_path.glob("*.sps")
    samples = read_samples(path)
    assert samples.hz == 500
    assert samples.regions == ["work"]


def test_samples_file_holds_regions_and_modules(tmp_path):
    run(build(tmp_path, PROGRAM, extra=("-g", "-O0")), tmp_path)

    (path,) = tmp_path.glob("*.sps")
    samples = read_samples(path)

    assert samples.rank == 0
    assert samples.hz == 1000
    assert samples.regions == ["hot", "cold"]
    assert len(samples.samples) > 50
    assert samples.modules
    assert set(samples.samples["region"]) == {0, 1}


@needs_symbolizer
def test_samples_land_in_the_function_that_ran(tmp_path):
    run(build(tmp_path, PROGRAM, extra=("-g", "-O0")), tmp_path)

    (path,) = tmp_path.glob("*.sps")
    profile = line_profile(read_samples(path))

    for region, function in (("hot", "hot_loop"), ("cold", "cold_loop")):
        top = profile[region][0]
        assert top.function == function
        assert top.file.endswith("prog.c")
        assert top.line in line_range(function)
        # Nothing of one region's samples may be attributed to the other's code.
        assert all(
            stat.line in line_range(function)
            for stat in profile[region]
            if stat.function != "??"
        )


@needs_symbolizer
def test_native_lines_command(tmp_path, capsys):
    run(build(tmp_path, PROGRAM, extra=("-g", "-O0")), tmp_path)

    main(["native-lines", str(tmp_path), "--top", "2", "--no-source"])

    out = capsys.readouterr().out
    assert "hot" in out
    assert "hot_loop" in out
    assert "cold_loop" in out


def test_reader_rejects_other_files(tmp_path):
    bogus = tmp_path / "x.sps"
    bogus.write_bytes(b"not a samples file")
    with pytest.raises(SampleFormatError):
        read_samples(bogus)

    real = tmp_path / "y.sps"
    run(build(tmp_path, PROGRAM, extra=("-g", "-O0")), tmp_path)
    (real_path,) = [p for p in tmp_path.glob("prof*.sps")]
    real.write_bytes(real_path.read_bytes()[:-5])
    with pytest.raises(SampleFormatError):
        read_samples(real)


def test_find_sample_files_reports_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_sample_files(tmp_path)
