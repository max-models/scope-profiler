"""How many bytes a profile costs, and how that scales.

File size is *deterministic* --- unlike the timings in ``test_overhead.py``, it
does not depend on machine load --- so the budgets here are tight rather than
an order of magnitude clear. That makes them real regression guards: a change
that stops packing small files, drops the delta encoding, or starts storing a
third int64 per call moves a number here immediately.

Payloads are synthetic and built from fixed arrays rather than recorded by
running a profile, so a rerun writes byte-identical files. Compression ratios
are measured on *jittered* data: constant gaps compress to almost nothing and
would flatter the encoding into a meaningless budget.

Every measurement is printed, plus a summary table once the module finishes.
pytest captures stdout, so pass ``-s`` to see it::

    pytest -s src/scope_profiler/tests/test_storage_size.py
"""

import h5py
import numpy as np
import pytest

from scope_profiler import load
from scope_profiler.h5writer import AUTO_COMPRESSION_MIN_EVENTS, ProfilingWriter
from scope_profiler.profile_manager import RankPayload

# A call is one int64 start and one int64 end. Everything above that --- the
# per-(rank, region) index row, the region name, the HDF5 object headers --- is
# overhead, and at scale it must round to nothing.
BYTES_PER_EVENT = 16

# What an event costs once the fixed cost is amortised away. Measured: 16.1-16.4
# bytes, i.e. the two int64 and almost nothing else. A third column per call, or
# a lost delta encoding, breaks this immediately.
MARGINAL_BYTES_BUDGET = 20

# The fixed cost of an HDF5 file with this schema's ~15 objects, none of which
# scales with the run. Measured: 15.7 KiB packed. Before publication packing it
# was 193 KiB, so this is the budget that guards the packing pass.
FLOOR_BYTES_BUDGET = 32 * 1024

# Compression on jittered data, which is the pessimistic case: real profiles
# compress better because their gaps are smaller and more regular. Measured:
# ~3.1-3.3 bytes/event, about 5x.
COMPRESSED_BYTES_BUDGET = 6
COMPRESSION_RATIO_FLOOR = 3.0

_MEASUREMENTS: list[tuple[str, float, str]] = []


def _report(label, value, unit, budget=None):
    """Print one measurement and keep it for the closing summary."""
    note = f"(budget {budget})" if budget is not None else ""
    _MEASUREMENTS.append((label, value, f"{unit} {note}".strip()))
    print(f"  {label:<44s} {value:10.1f} {unit} {note}")


@pytest.fixture(scope="module", autouse=True)
def size_summary():
    """Print every measurement in one table once the module has finished."""
    yield
    if not _MEASUREMENTS:
        return
    width = max(len(label) for label, _, _ in _MEASUREMENTS)
    print("\n\nOutput file size")
    print("-" * (width + 34))
    for label, value, unit in _MEASUREMENTS:
        print(f"{label:<{width}s}  {value:10.1f} {unit}")
    print("-" * (width + 34))


def _payload(regions, events, *, jitter=False, seed=0):
    """One rank's arrays, deterministic for a given seed.

    Without ``jitter`` the gaps and durations are constant, which is the best
    case for the delta encoding and therefore not what a ratio should be
    measured on.
    """
    generator = np.random.default_rng(seed)
    recorded = {}
    for index in range(regions):
        if jitter:
            gaps = generator.integers(300, 1500, events)
            durations = generator.integers(50, 400, events)
        else:
            gaps = np.full(events, 1000)
            durations = np.full(events, 250)
        starts = (np.cumsum(gaps) + index * 10**9).astype(np.int64)
        recorded[f"region_{index}"] = (starts, (starts + durations).astype(np.int64))
    return RankPayload(regions=recorded, likwid={}, likwid_environment={})


def _write(path, regions, events, *, ranks=1, jitter=False, **options):
    """Write a profile and return its size in bytes."""
    with ProfilingWriter(path, {"hostname": "node"}, repack=True, **options) as writer:
        for rank in range(ranks):
            writer.write_rank(rank, _payload(regions, events, jitter=jitter, seed=rank))
    return path.stat().st_size


# --- the fixed cost ---------------------------------------------------------


def test_a_minimal_profile_fits_in_the_floor(tmp_path):
    """One region, one call: all overhead, and it must stay bounded."""
    size = _write(tmp_path / "floor.h5", 1, 1)

    _report(
        "floor: 1 region, 1 call",
        size / 1024,
        "KiB",
        f"{FLOOR_BYTES_BUDGET // 1024} KiB",
    )
    assert size < FLOOR_BYTES_BUDGET


def test_the_floor_does_not_depend_on_the_region_count(tmp_path):
    """Ten empty-ish regions cost a little more than one, not ten times more."""
    one = _write(tmp_path / "one.h5", 1, 1)
    ten = _write(tmp_path / "ten.h5", 10, 1)

    _report("floor: 10 regions vs 1 region", ten / one, "x")
    assert ten < one * 2


# --- the marginal cost ------------------------------------------------------


@pytest.mark.parametrize("events", [10_000, 100_000])
def test_an_event_costs_about_two_int64(tmp_path, events):
    """At scale the file is its two timestamp columns and nothing else."""
    size = _write(tmp_path / f"scale_{events}.h5", 1, events)
    per_event = size / events

    _report(
        f"uncompressed, {events:>7} events", per_event, "B/event", MARGINAL_BYTES_BUDGET
    )
    assert per_event < MARGINAL_BYTES_BUDGET
    # ...and never below the information actually stored.
    assert per_event > BYTES_PER_EVENT * 0.5


def test_size_grows_linearly_with_the_event_count(tmp_path):
    """The marginal byte cost is flat: no per-event index, no quadratic term."""
    small = _write(tmp_path / "lin_small.h5", 1, 50_000)
    large = _write(tmp_path / "lin_large.h5", 1, 100_000)
    marginal = (large - small) / 50_000

    _report(
        "marginal cost of 50k more events", marginal, "B/event", MARGINAL_BYTES_BUDGET
    )
    assert marginal < MARGINAL_BYTES_BUDGET
    # Doubling the events must not much more than double the file.
    assert large < small * 2.1


def test_size_grows_linearly_with_the_rank_count(tmp_path):
    """Each rank adds its own events and one index row per region, no more."""
    one = _write(tmp_path / "rank1.h5", 4, 5_000, ranks=1)
    eight = _write(tmp_path / "rank8.h5", 4, 5_000, ranks=8)

    _report("8 ranks vs 1 rank, same per-rank work", eight / one, "x")
    assert eight < one * 8.5
    _report(
        "per-event cost at 8 ranks",
        eight / (8 * 4 * 5_000),
        "B/event",
        MARGINAL_BYTES_BUDGET,
    )
    assert eight / (8 * 4 * 5_000) < MARGINAL_BYTES_BUDGET


def test_many_regions_cost_an_index_row_not_a_file_each(tmp_path):
    """Region count drives the index, which is per (rank, region), not per call."""
    few = _write(tmp_path / "few.h5", 2, 5_000)
    many = _write(tmp_path / "many.h5", 200, 50)
    # Same 10,000 events either way; 200 regions cost 200 index rows and names.
    _report("200 regions vs 2, same event count", many / few, "x")
    assert many < few * 2.5


# --- compression ------------------------------------------------------------


def test_auto_compression_earns_its_place_on_jittered_data(tmp_path):
    """Measured on the pessimistic case: real profiles compress better."""
    events = AUTO_COMPRESSION_MIN_EVENTS * 4
    plain = _write(tmp_path / "plain.h5", 4, events, jitter=True)
    packed = _write(tmp_path / "auto.h5", 4, events, jitter=True, compression="auto")
    total = 4 * events

    _report("compressed", packed / total, "B/event", COMPRESSED_BYTES_BUDGET)
    _report(
        "compression ratio (jittered)",
        plain / packed,
        "x",
        f">{COMPRESSION_RATIO_FLOOR}",
    )
    assert packed / total < COMPRESSED_BYTES_BUDGET
    assert plain / packed > COMPRESSION_RATIO_FLOOR


def test_compression_does_not_change_what_the_file_says(tmp_path):
    """The whole analysis is worthless if the bytes stop round-tripping."""
    events = AUTO_COMPRESSION_MIN_EVENTS + 1
    plain = tmp_path / "plain.h5"
    packed = tmp_path / "auto.h5"
    _write(plain, 2, events, jitter=True)
    _write(packed, 2, events, jitter=True, compression="auto")

    left, right = load(plain), load(packed)
    assert left.region_names == right.region_names
    for name in left.region_names:
        assert (
            left[name][0].start_times_ns.tolist()
            == right[name][0].start_times_ns.tolist()
        )
        assert (
            left[name][0].end_times_ns.tolist() == right[name][0].end_times_ns.tolist()
        )


# --- the encoding earns its place ------------------------------------------


def test_delta_encoding_beats_absolute_timestamps_at_the_same_filter(tmp_path):
    """The schema-3 encoding, measured against what schema 2 would have stored.

    Both columns hold the same information and go through the same gzip and
    shuffle filters, so the difference is the encoding alone. Jittered data
    again, which is the encoding's worst case.
    """
    events = 50_000
    path = tmp_path / "encoded.h5"
    _write(path, 4, events, jitter=True, compression="auto")

    with h5py.File(path, "r") as handle:
        deltas = handle["events/start_deltas"][()]
        durations = handle["events/durations"][()]
        offsets = handle["rank_region_index/event_offsets"][()]
        counts = handle["rank_region_index/event_counts"][()]

    absolute = deltas.copy()
    for offset, count in zip(offsets, counts):
        offset, count = int(offset), int(count)
        if count:
            absolute[offset : offset + count] = np.cumsum(
                deltas[offset : offset + count],
            )
    ends = absolute + durations

    filters = {"compression": "gzip", "compression_opts": 4, "shuffle": True}

    def stored(name, columns):
        target = tmp_path / name
        with h5py.File(target, "w") as handle:
            for key, values in columns.items():
                handle.create_dataset(key, data=values, **filters)
        return target.stat().st_size

    schema_two = stored("absolute.h5", {"start": absolute, "end": ends})
    schema_three = stored("delta.h5", {"start": deltas, "duration": durations})

    _report("delta vs absolute, same filter", schema_two / schema_three, "x", ">1.2")
    assert schema_three < schema_two


# --- aggregation mode -------------------------------------------------------


def test_aggregation_mode_size_does_not_grow_with_the_call_count(tmp_path):
    """No timeline is stored, so a billion calls cost what ten do."""
    from scope_profiler import ProfileManager

    sizes = []
    for calls in (10, 10_000):
        path = tmp_path / f"agg_{calls}.h5"
        with ProfileManager.session(
            file_path=str(path),
            aggregation_mode=True,
            verbose=False,
        ):
            region = ProfileManager.region("hot")
            for _ in range(calls):
                with region:
                    pass
        sizes.append(path.stat().st_size)
        assert load(path)["hot"][0].num_calls == calls
    ProfileManager._reset()

    _report("aggregation: 10k calls vs 10 calls", sizes[1] / sizes[0], "x", "~1.0")
    assert sizes[1] == pytest.approx(sizes[0], rel=0.05)
    assert sizes[1] < FLOOR_BYTES_BUDGET
