"""The rank-0 receive loop, driven by a stand-in communicator.

``finalize()``'s multi-rank path is otherwise only reachable under ``mpirun``,
which pytest is never started by. A stand-in communicator exercises it in
process: it is enough to pin down the ordering, the one-payload-at-a-time
discipline, and the "silent rank gets no group" rule, none of which a
single-rank run can show.
"""

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from scope_profiler import ProfileManager, read_h5
from scope_profiler.profile_manager import RankPayload

NS = 1_000_000_000


class FakeComm:
    """Just enough of an mpi4py communicator for the receive loop.

    Payloads for the other ranks are queued up front; ``recv`` hands out the
    one for the rank it was asked for, so the loop's receive order is
    observable in ``recv_order``.
    """

    def __init__(self, rank: int, size: int, payloads: dict | None = None) -> None:
        self.rank = rank
        self.size = size
        self._payloads = payloads or {}
        self.recv_order: list = []
        self.sent: list = []
        self.barriers = 0

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.size

    def send(self, payload, dest, tag=0) -> None:
        self.sent.append((dest, payload))

    def recv(self, source, tag=0):
        self.recv_order.append(source)
        return self._payloads[source]

    def Barrier(self) -> None:
        self.barriers += 1


def payload(*durations_ns, name="solve") -> RankPayload:
    """One region entered once per duration given."""
    starts = np.arange(len(durations_ns), dtype=np.int64) * 10 * NS
    ends = starts + np.asarray(durations_ns, dtype=np.int64)
    return RankPayload(regions={name: (starts, ends)}, likwid={}, likwid_environment={})


@pytest.fixture
def configured(tmp_path):
    """A configured manager whose config can be pointed at a fake comm."""
    ProfileManager.setup(file_path=str(tmp_path / "fake.h5"))
    yield ProfileManager.get_config()
    ProfileManager._reset()


def test_rank0_receives_every_other_rank_in_rank_order(configured, tmp_path):
    remote = {rank: payload(rank * NS) for rank in range(1, 4)}
    comm = FakeComm(rank=0, size=4, payloads=remote)
    configured._comm = comm
    configured._rank, configured._size = 0, 4

    results = ProfileManager._collect_payloads(
        payload(9 * NS), write_file=True, need_results=True
    )

    # Every remote rank is received exactly once, in rank order: pooled
    # statistics depend on it, and so does agreement with the file.
    assert comm.recv_order == [1, 2, 3]

    assert list(results["solve"].regions) == [0, 1, 2, 3]
    assert results["solve"].num_calls == 4
    # ...and the file that was written alongside agrees, region for region.
    assert results.summary() == read_h5(configured.file_path).summary()


def test_a_silent_rank_yields_no_group_but_still_sends(configured, tmp_path):
    """The empty payload must travel; only its group is skipped."""
    remote = {
        1: RankPayload(regions={}, likwid={}, likwid_environment={}),
        2: payload(2 * NS),
    }
    comm = FakeComm(rank=0, size=3, payloads=remote)
    configured._comm = comm
    configured._rank, configured._size = 0, 3

    results = ProfileManager._collect_payloads(
        payload(NS), write_file=True, need_results=True
    )

    # Rank 1 was still received -- skipping the receive would deadlock.
    assert comm.recv_order == [1, 2]
    with h5py.File(configured.file_path, "r") as handle:
        assert handle["rank_region_index/ranks"][()].tolist() == [0, 2]
    assert list(results["solve"].regions) == [0, 2]


def test_non_root_sends_once_and_gets_empty_results(configured):
    comm = FakeComm(rank=2, size=4)
    configured._comm = comm
    configured._rank, configured._size = 2, 4

    results = ProfileManager._collect_payloads(
        payload(NS), write_file=True, need_results=True
    )

    assert [dest for dest, _ in comm.sent] == [0]
    assert not results.is_root
    assert results.region_names == []
    # A non-root rank writes nothing at all.
    assert not os.path.exists(configured.file_path)


def test_results_without_a_file_still_stream(configured):
    comm = FakeComm(rank=0, size=2, payloads={1: payload(3 * NS)})
    configured._comm = comm
    configured._rank, configured._size = 0, 2

    results = ProfileManager._collect_payloads(
        payload(NS), write_file=False, need_results=True
    )

    assert comm.recv_order == [1]
    assert results["solve"].num_calls == 2
    assert not os.path.exists(configured.file_path)


def test_the_output_file_is_closed_even_on_error(configured, monkeypatch):
    """A write failure preserves the previous file and removes its temporary."""
    comm = FakeComm(rank=0, size=2, payloads={1: payload(NS)})
    configured._comm = comm
    configured._rank, configured._size = 0, 2

    from scope_profiler import h5writer

    with h5writer.ProfilingWriter(configured.file_path, {"previous": "profile"}):
        pass

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(h5writer.ProfilingWriter, "write_rank", boom)

    with pytest.raises(OSError):
        ProfileManager._collect_payloads(
            payload(NS), write_file=True, need_results=False
        )
    # The destination still contains the complete previous run, not a partial
    # metadata-only replacement from the failed write.
    with h5py.File(configured.file_path, "r") as handle:
        assert handle["metadata"].attrs["previous"] == "profile"
    assert list(Path(configured.file_path).parent.glob(".*.tmp")) == []


def test_no_collective_is_used_for_the_transport(configured):
    """Only point-to-point calls, so a forked rank cannot crash on a collective.

    ``use_likwid=True`` reads its counters back in a subprocess, and Open MPI
    does not support forking from a rank on its shared-memory transport --
    ``MPI_Comm_dup`` afterwards segfaults. A stand-in communicator that offers
    nothing but send/recv proves this path never needs more than that.
    """

    class PointToPointOnly(FakeComm):
        def Dup(self):
            raise AssertionError("the transport must not duplicate the communicator")

        def Barrier(self):
            raise AssertionError("the transport must not use a collective")

        def gather(self, *args, **kwargs):
            raise AssertionError("the transport must not use a collective")

    comm = PointToPointOnly(rank=0, size=2, payloads={1: payload(NS)})
    configured._comm = comm
    configured._rank, configured._size = 0, 2

    results = ProfileManager._collect_payloads(
        payload(NS), write_file=True, need_results=True
    )
    assert results["solve"].num_calls == 2


def test_direct_writer_appends_a_rank_after_receiving_the_token(configured, tmp_path):
    """A non-root rank writes its own arrays; no payload is sent to rank 0."""
    from scope_profiler.h5writer import ProfilingWriter

    temp_path = configured.file_path + ".scope-profiler.tmp"
    with ProfilingWriter(temp_path, configured.metadata) as writer:
        writer.write_rank(0, payload(NS))

    comm = FakeComm(rank=1, size=2, payloads={0: (True, "")})
    configured._comm = comm
    configured._rank, configured._size = 1, 2

    ProfileManager._write_payload_direct(payload(2 * NS, name="remote"))

    assert comm.sent == [(0, (True, ""))]
    with h5py.File(temp_path, "r") as handle:
        assert handle["region_table/names"][()].tolist() == [b"solve", b"remote"]
        assert handle["rank_region_index/ranks"][()].tolist() == [0, 1]


def test_direct_writer_publishes_single_rank_file_atomically(configured):
    configured._rank, configured._size = 0, 1

    ProfileManager._write_payload_direct(payload(NS))

    assert os.path.exists(configured.file_path)
    assert not os.path.exists(configured.file_path + ".scope-profiler.tmp")
    assert read_h5(configured.file_path)["solve"].num_calls == 1


def test_direct_writer_forwards_an_existing_failure_without_opening_file(
    configured,
):
    comm = FakeComm(rank=1, size=3, payloads={0: (False, "rank 0 failed")})
    configured._comm = comm
    configured._rank, configured._size = 1, 3

    ProfileManager._write_payload_direct(payload(NS))

    assert comm.sent == [(2, (False, "rank 0 failed"))]
    assert not os.path.exists(configured.file_path + ".scope-profiler.tmp")


def test_auto_prefers_parallel_hdf5_when_available(configured, monkeypatch):
    from scope_profiler import h5writer

    comm = FakeComm(rank=0, size=2)
    configured._comm = comm
    configured._rank, configured._size = 0, 2
    calls = []
    monkeypatch.setattr(h5writer, "parallel_hdf5_available", lambda: True)
    monkeypatch.setattr(
        h5writer,
        "write_parallel_payload",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(h5writer, "atomic_publish", lambda *args: calls.append(args))

    ProfileManager._write_payload_file(payload(NS))

    assert len(calls) == 2
    assert calls[0][0][1] is comm
    assert comm.barriers == 2


def test_explicit_parallel_mode_requires_mpi_enabled_h5py(configured, monkeypatch):
    from scope_profiler import h5writer

    configured._output_mode = "parallel"
    monkeypatch.setattr(h5writer, "parallel_hdf5_available", lambda: False)

    with pytest.raises(RuntimeError, match="h5py build with MPI support"):
        ProfileManager._write_payload_file(payload(NS))
