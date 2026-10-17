"""Line-level profiles of native (C) regions, from the sampler's ``.sps`` files.

With sampling on (``sp_start_sampling()`` or ``SCOPE_PROFILER_SAMPLE_HZ``), the
C library records the interrupted instruction address and the innermost open
region at every timer tick, and ``sp_finalize()`` writes them to
``<prefix>_rank<NNNNN>.sps`` next to the trace. This module reads that file and
maps the addresses to ``file:line`` using the program's debug info (build with
``-g``), so the work happens after the run, not in it.

Layout, native byte order, as written by ``sp_finalize``::

    char[8]   "SCOPESMP"
    int32     format version (1)
    int32     rank
    int32     sampling rate [Hz]
    int64     timer ticks in total, whether or not they landed in a region
    int32     number of regions
    per region:  int32 name length, char[] name
    int32     number of modules
    per module:  int32 path length, char[] path, uint64 load address
    int64     number of samples
    per sample:  int32 region, int32 module (-1 if unknown), uint64 address
"""

import platform
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"SCOPESMP"
SAMPLES_SUFFIX = ".sps"
KNOWN_FORMAT_VERSIONS = (1,)

_SAMPLE = np.dtype([("region", "i4"), ("module", "i4"), ("ip", "u8")])


class SampleFormatError(ValueError):
    """A file is not a scope-profiler sample file, or is truncated."""


@dataclass
class SampleFile:
    """One rank's raw samples."""

    rank: int
    hz: int
    ticks: int
    regions: list
    modules: list  # (path, load address)
    samples: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class LineStat:
    """Samples attributed to one source line of one region."""

    file: str
    line: int
    function: str
    samples: int


def read_samples(path) -> SampleFile:
    """Read one ``.sps`` file written by ``sp_finalize()``.

    Raises
    ------
    SampleFormatError
        If the file is not a sample file, is truncated, or is of a newer
        format version.
    """
    path = Path(path)
    buffer = path.read_bytes()
    if buffer[:8] != MAGIC:
        raise SampleFormatError(f"{path}: not a scope-profiler sample file")

    offset = 8

    def take(fmt):
        nonlocal offset
        dtype = np.dtype(fmt)
        if offset + dtype.itemsize > len(buffer):
            raise SampleFormatError(f"{path}: truncated sample file")
        value = np.frombuffer(buffer, dtype=dtype, count=1, offset=offset)[0]
        offset += dtype.itemsize
        return value

    def take_str():
        nonlocal offset
        length = int(take("i4"))
        if length < 0 or offset + length > len(buffer):
            raise SampleFormatError(f"{path}: truncated sample file")
        text = buffer[offset : offset + length].decode("utf-8", "replace")
        offset += length
        return text

    version = int(take("i4"))
    if version not in KNOWN_FORMAT_VERSIONS:
        raise SampleFormatError(
            f"{path}: unsupported sample format version {version} "
            f"(this scope-profiler reads versions {KNOWN_FORMAT_VERSIONS})",
        )
    rank = int(take("i4"))
    hz = int(take("i4"))
    ticks = int(take("i8"))
    regions = [take_str() for _ in range(int(take("i4")))]
    modules = []
    for _ in range(int(take("i4"))):
        module_path = take_str()
        modules.append((module_path, int(take("u8"))))

    count = int(take("i8"))
    end = offset + count * _SAMPLE.itemsize
    if end != len(buffer):
        raise SampleFormatError(f"{path}: truncated or oversized sample file")
    samples = np.frombuffer(buffer, dtype=_SAMPLE, count=count, offset=offset).copy()
    return SampleFile(rank, hz, ticks, regions, modules, samples)


def find_sample_files(inputs) -> list:
    """Collect ``.sps`` files from paths, directories, or a mix of both."""
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    found = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            in_dir = sorted(path.glob(f"*{SAMPLES_SUFFIX}"))
            if not in_dir:
                raise FileNotFoundError(f"no {SAMPLES_SUFFIX} files in {path}")
            found.extend(in_dir)
        elif path.exists():
            found.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    return sorted(set(found))


class SymbolizerError(RuntimeError):
    """No tool is available to map addresses to source lines."""


def _symbolize_atos(module: str, base: int, addresses: list) -> list:
    """``atos`` reads the debug map of the executable, so no dSYM is needed."""
    command = ["atos", "-o", module, "-l", hex(base), *(hex(a) for a in addresses)]
    out = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    # "func (in module) (file.c:12)"; without line info only "func (in module)".
    results = []
    for text in out.splitlines():
        match = re.match(r"^(.*?) \(in .*?\)(?: \((.*):(\d+)\))?\s*$", text)
        if match is None:
            results.append(("??", "??", 0))
        else:
            function, file, line = match.groups()
            results.append((function, file or "??", int(line or 0)))
    return results


def _symbolize_addr2line(module: str, base: int, addresses: list) -> list:
    """GNU ``addr2line``/``llvm-symbolizer`` take offsets into the module."""
    tool = shutil.which("llvm-symbolizer") or shutil.which("addr2line")
    if tool is None:
        raise SymbolizerError("neither llvm-symbolizer nor addr2line is installed")
    # Position-independent code (the default for shared libraries and most
    # executables today) is addressed relative to its load address. A fixed
    # address executable would need the absolute address instead.
    offsets = [hex(a - base) for a in addresses]
    if tool.endswith("llvm-symbolizer"):
        command = [tool, "--functions=linkage", "--demangle", "-e", module, *offsets]
        out = subprocess.run(command, capture_output=True, text=True, check=True).stdout
        blocks = out.strip().split("\n\n")
        results = []
        for block in blocks:
            lines = block.splitlines()
            function = lines[0] if lines else "??"
            where = lines[1] if len(lines) > 1 else "??:0"
            file, _, line = where.rpartition(":")
            results.append((function, file, int(line.split(":")[0] or 0)))
        return results
    command = [tool, "-f", "-C", "-e", module, *offsets]
    out = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    results = []
    for function, where in zip(lines[0::2], lines[1::2]):
        file, _, line = where.rpartition(":")
        digits = re.match(r"\d+", line)
        results.append((function, file, int(digits.group()) if digits else 0))
    return results


def symbolize(module: str, base: int, addresses: list) -> list:
    """Map absolute addresses in one module to ``(function, file, line)``.

    Needs the program built with ``-g``; without debug info the line is 0.
    """
    unknown = [("??", "??", 0)] * len(addresses)
    # Pseudo-modules such as linux-vdso.so.1 have no file to read debug info from.
    if not addresses or not Path(module).exists():
        return unknown
    try:
        if platform.system() == "Darwin" and shutil.which("atos"):
            return _symbolize_atos(module, base, addresses)
        return _symbolize_addr2line(module, base, addresses)
    except subprocess.CalledProcessError:
        return unknown


def line_profile(samples: SampleFile) -> dict:
    """Aggregate a rank's samples into per-line statistics for each region.

    Returns
    -------
    dict
        ``{region name: [LineStat, ...]}``, hottest line first. Counts are
        samples of the line itself; code without debug info (a library such as
        libm) is grouped under ``??:0``. The timer's effective rate can be
        below the requested one, so shares are reported rather than seconds.
    """
    by_module: dict = {}
    for module_index, (path, base) in enumerate(samples.modules):
        ips = np.unique(
            samples.samples["ip"][samples.samples["module"] == module_index]
        )
        if ips.size:
            addresses = [int(ip) for ip in ips]
            by_module[module_index] = dict(
                zip(addresses, symbolize(path, base, addresses)),
            )

    counts: dict = {}
    for region, module, ip in samples.samples:
        if region < 0 or region >= len(samples.regions):
            continue
        place = by_module.get(int(module), {}).get(int(ip), ("??", "??", 0))
        counts.setdefault(samples.regions[region], Counter())[place] += 1

    return {
        region: [
            LineStat(file, line, function, n)
            for (function, file, line), n in counter.most_common()
        ]
        for region, counter in counts.items()
    }


def format_line_profile(profile: dict, top: int = 10, source: bool = True) -> str:
    """Render :func:`line_profile` output as plain-text tables."""
    import linecache

    from tabulate import tabulate

    blocks = []
    for region, stats in profile.items():
        total = sum(s.samples for s in stats) or 1
        rows = []
        for stat in stats[:top]:
            text = ""
            if source and stat.line > 0:
                text = linecache.getline(stat.file, stat.line).strip()
            rows.append(
                [
                    f"{stat.file.rsplit('/', 1)[-1]}:{stat.line}",
                    stat.function,
                    stat.samples,
                    f"{100 * stat.samples / total:5.1f}%",
                    text,
                ],
            )
        headers = ["line", "function", "samples", "% region", "source"]
        if not source:
            rows = [r[:-1] for r in rows]
            headers = headers[:-1]
        blocks.append(
            f"{region}  ({total} samples)\n"
            + tabulate(rows, headers=headers, tablefmt="simple"),
        )
    return "\n\n".join(blocks)
