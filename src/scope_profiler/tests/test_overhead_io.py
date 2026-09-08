"""Cost of writing a profile out and reading it back.

``test_overhead.py`` covers the per-call instrumentation cost --- what a
profiled region adds to the program being measured. This module covers the
other half: what ``finalize()`` spends turning the recorded buffers into a
file, and what post-processing spends reading one back.

Both matter for different reasons. The write happens inside the profiled
program, at the end of a run that may have been queued for hours, and under
MPI it is collective --- so a write that scales badly in the event count
stalls every rank. The read happens in analysis, where a profile is opened
repeatedly.

As in ``test_overhead.py`` the budgets are regression guards, not benchmarks:
they sit roughly an order of magnitude above what an idle laptop measures, so
a loaded CI machine still passes while a structural regression --- a per-event
Python loop, a quadratic index rebuild, a repack of a file far too large for
one --- does not. Each measurement takes the *minimum* over repeats, which is
the robust estimator for a cost.

The scaling assertions matter more than the absolute ones. A budget can only
catch a change big enough to blow it; a ratio between two sizes of the same
measurement catches a change in the *shape* of the cost, which is what
actually goes wrong here.

Every measurement is printed, plus a summary table once the module finishes::

    pytest -s -m overhead src/scope_profiler/tests/test_overhead_io.py
"""

import math
import shutil
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pytest

from scope_profiler import load, read_h5_summary
from scope_profiler.h5writer import (
    AUTO_COMPRESSION_MIN_EVENTS,
    REPACK_MAX_FILE_BYTES,
    ProfilingWriter,
    decode_start_deltas,
    encode_start_deltas,
    publish_file,
)
from scope_profiler.profile_manager import RankPayload

# Timing sensitive, and it writes multi-megabyte files: deselect with
# '-m "not overhead"'.
pytestmark = [pytest.mark.overhead, pytest.mark.slow]

REPEATS = 3

# Nanoseconds per event to write a profile out, including the publication
# pass. Measured on an idle laptop: ~93 ns/event at 100k events, but ~510 at
# 10k -- the publication copy is a fixed ~2.5 ms, so a small profile is
# dominated by it and gets its own budget rather than one blurry number that
# would be loose at scale and tight below it.
WRITE_BUDGET_NS = {10_000: 8_000, 100_000: 2_000}

# Nanoseconds per event to read one back. Measured: ~10-25 ns/event
# uncompressed, ~60-70 ns/event through gzip.
READ_BUDGET_NS = 1_000
READ_COMPRESSED_BUDGET_NS = 2_000

# The publication pass copies a small file object by object. Measured: ~2.3 ms.
PUBLISH_BUDGET_NS = 200_000_000

# Encoding and decoding are one vectorised pass over a column.
CODEC_BUDGET_NS = 100

_MEASUREMENTS: list[tuple[str, float, str, float | None]] = []


def _report(label, value, unit, budget=None):
    """Print one measurement and keep it for the closing summary."""
    _MEASUREMENTS.append((label, value, unit, budget))
    if budget is None:
        print(f"  {label:<44s} {value:10.1f} {unit}")
        return
    headroom = budget / value if value > 0 else math.inf
    print(
        f"  {label:<44s} {value:10.1f} {unit}   "
        f"(budget {budget:.0f}, {headroom:5.1f}x headroom)",
    )


@pytest.fixture(scope="module", autouse=True)
def io_summary():
    """Print every measurement in one table once the module has finished."""
    yield
    if not _MEASUREMENTS:
        return
    width = max(len(label) for label, *_ in _MEASUREMENTS)
    print("\n\nProfile write and read cost")
    print("-" * (width + 44))
    for label, value, unit, budget in _MEASUREMENTS:
        tail = "" if budget is None else f"   budget {budget:>12.0f}"
        print(f"{label:<{width}s}  {value:10.1f} {unit}{tail}")
    print("-" * (width + 44))


def _payload(regions, events, seed=0):
    """One rank's arrays, jittered so compression is not flattered."""
    generator = np.random.default_rng(seed)
    recorded = {}
    for index in range(regions):
        gaps = generator.integers(300, 1500, events)
        starts = (np.cumsum(gaps) + index * 10**9).astype(np.int64)
        durations = generator.integers(50, 400, events)
        recorded[f"region_{index}"] = (starts, (starts + durations).astype(np.int64))
    return RankPayload(regions=recorded, likwid={}, likwid_environment={})


def _best_ns(body, repeats=REPEATS):
    """Fastest observed run of ``body``, in nanoseconds."""
    best = math.inf
    for _ in range(repeats):
        start = perf_counter_ns()
        body()
        best = min(best, perf_counter_ns() - start)
    return best


def _write_ns(path, regions, events, *, ranks=1, **options):
    """Nanoseconds to write this profile, taking the best of several runs."""
    payloads = [_payload(regions, events, seed=rank) for rank in range(ranks)]

    def body():
        Path(path).unlink(missing_ok=True)
        with ProfilingWriter(path, {"h": "n"}, repack=True, **options) as writer:
            for rank, payload in enumerate(payloads):
                writer.write_rank(rank, payload)

    return _best_ns(body)


# --- writing ----------------------------------------------------------------


@pytest.mark.parametrize("events", [10_000, 100_000])
def test_writing_costs_a_bounded_amount_per_event(tmp_path, events):
    elapsed = _write_ns(tmp_path / f"w{events}.h5", 4, events // 4)

    per_event = elapsed / events
    budget = WRITE_BUDGET_NS[events]
    _report(f"write, {events:>7} events", per_event, "ns/event", budget)
    assert per_event < budget


def test_write_cost_scales_linearly_with_events(tmp_path):
    """Ten times the events must not cost far more than ten times the work.

    This is the assertion that catches a per-event Python loop or a quadratic
    index rebuild, which no absolute budget reliably would.
    """
    small = _write_ns(tmp_path / "ws.h5", 4, 2_500)
    large = _write_ns(tmp_path / "wl.h5", 4, 25_000)

    growth = large / small
    _report("write: 10x events costs", growth, "x", 30)
    assert growth < 30


def test_auto_compression_does_not_dominate_the_write(tmp_path):
    """The filter costs write CPU; it must stay the same order of magnitude."""
    events = AUTO_COMPRESSION_MIN_EVENTS * 2
    plain = _write_ns(tmp_path / "p.h5", 4, events // 4)
    packed = _write_ns(tmp_path / "a.h5", 4, events // 4, compression="auto")

    ratio = packed / plain
    _report("auto compression, write cost", ratio, "x plain", 10)
    assert ratio < 10


# --- publication ------------------------------------------------------------


def test_publishing_a_small_profile_is_cheap(tmp_path):
    """The packing pass copies the file, so its cost must stay small."""
    source = tmp_path / "src.h5"
    with ProfilingWriter(source, {"h": "n"}) as writer:
        writer.write_rank(0, _payload(5, 20))

    def body():
        target = tmp_path / "copy.h5"
        shutil.copy(source, target)
        publish_file(target)

    elapsed = _best_ns(body)
    _report("publish a small profile", elapsed / 1e6, "ms", PUBLISH_BUDGET_NS / 1e6)
    assert elapsed < PUBLISH_BUDGET_NS


def test_a_large_uncompressed_profile_is_not_repacked(tmp_path):
    """Above the size gate the copy would cost more than the space it saves.

    Asserted structurally rather than by timing: ``publish_file`` reports
    whether it rewrote the file, which is the decision under test.
    """
    path = tmp_path / "large.h5"
    with ProfilingWriter(path, {"h": "n"}, repack=True) as writer:
        # Comfortably past REPACK_MAX_FILE_BYTES at 16 bytes an event.
        writer.write_rank(0, _payload(4, REPACK_MAX_FILE_BYTES // 16))
    assert path.stat().st_size > REPACK_MAX_FILE_BYTES

    assert publish_file(path) is False


def test_publishing_is_idempotent(tmp_path):
    """A published file is already in its final layout; publishing again is a no-op."""
    path = tmp_path / "twice.h5"
    with ProfilingWriter(path, {"h": "n"}, repack=True) as writer:
        writer.write_rank(0, _payload(3, 40))
    first = path.stat().st_size

    publish_file(path)

    assert path.stat().st_size == first
    assert load(path)["region_0"][0].num_calls == 40


# --- reading ----------------------------------------------------------------


@pytest.mark.parametrize("compression", [None, "auto"])
def test_reading_costs_a_bounded_amount_per_event(tmp_path, compression):
    events = 100_000
    path = tmp_path / f"r_{compression}.h5"
    options = {"compression": compression} if compression else {}
    with ProfilingWriter(path, {"h": "n"}, repack=True, **options) as writer:
        writer.write_rank(0, _payload(4, events // 4))

    elapsed = _best_ns(lambda: load(path))
    per_event = elapsed / events
    budget = READ_COMPRESSED_BUDGET_NS if compression else READ_BUDGET_NS

    _report(f"read, {compression or 'plain':>5}", per_event, "ns/event", budget)
    assert per_event < budget


def test_read_cost_scales_linearly_with_events(tmp_path):
    """The decode is per (rank, region) run, not per event in Python."""

    def build(name, events):
        path = tmp_path / name
        with ProfilingWriter(path, {"h": "n"}, repack=True) as writer:
            writer.write_rank(0, _payload(4, events // 4))
        return path

    small = build("rs.h5", 20_000)
    large = build("rl.h5", 200_000)
    growth = _best_ns(lambda: load(large)) / _best_ns(lambda: load(small))

    _report("read: 10x events costs", growth, "x", 30)
    assert growth < 30


def test_the_decode_does_not_scale_with_the_region_count(tmp_path):
    """One cumsum per run, so many small runs must not become the cost."""

    def build(name, regions, events):
        path = tmp_path / name
        with ProfilingWriter(path, {"h": "n"}, repack=True) as writer:
            writer.write_rank(0, _payload(regions, events))
        return path

    few = build("few.h5", 4, 5_000)
    many = build("many.h5", 200, 100)
    ratio = _best_ns(lambda: load(many)) / _best_ns(lambda: load(few))

    _report("read: 200 runs vs 4, same events", ratio, "x", 40)
    assert ratio < 40


def test_a_summary_only_read_does_not_scale_with_the_event_count(tmp_path):
    """Its whole purpose: the event columns are never touched.

    A ten-times-larger profile must cost about the same, which is what proves
    the summary path really is reading only the fixed-size index.
    """

    def build(name, events):
        path = tmp_path / name
        with ProfilingWriter(path, {"h": "n"}, repack=True) as writer:
            writer.write_rank(0, _payload(4, events // 4))
        return path

    small = build("ss.h5", 20_000)
    large = build("sl.h5", 200_000)
    ratio = _best_ns(lambda: read_h5_summary(large)) / _best_ns(
        lambda: read_h5_summary(small),
    )

    _report("summary read: 10x events costs", ratio, "x", 4)
    assert ratio < 4
    # ...and it is genuinely cheaper than the full read it stands in for.
    assert _best_ns(lambda: read_h5_summary(large)) < _best_ns(lambda: load(large))


# --- the codec itself -------------------------------------------------------


def test_the_codec_is_one_vectorised_pass(tmp_path):
    """Encode and decode are numpy calls, not per-event Python."""
    starts = np.cumsum(np.random.default_rng(0).integers(1, 1000, 1_000_000))
    starts = starts.astype(np.int64)

    encode = _best_ns(lambda: encode_start_deltas(starts)) / starts.size
    deltas = encode_start_deltas(starts)
    decode = _best_ns(lambda: decode_start_deltas(deltas)) / deltas.size

    _report("encode_start_deltas", encode, "ns/event", CODEC_BUDGET_NS)
    _report("decode_start_deltas", decode, "ns/event", CODEC_BUDGET_NS)
    assert encode < CODEC_BUDGET_NS
    assert decode < CODEC_BUDGET_NS
    assert decode_start_deltas(deltas).tolist() == starts.tolist()
