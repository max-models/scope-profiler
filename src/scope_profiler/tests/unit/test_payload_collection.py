"""The rank-0 receive loop, driven by a stand-in communicator.

``finalize()``'s multi-rank path is otherwise only reachable under ``mpirun``,
which pytest is never started by. A stand-in communicator exercises it in
process: it is enough to pin down the ordering, the one-payload-at-a-time
discipline, and the "silent rank gets no group" rule, none of which a
single-rank run can show.
"""

import os

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

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.size

    def send(self, payload, dest, tag=0) -> None:
        self.sent.append((dest, payload))

    def recv(self, source, tag=0):
        self.recv_order.append(source)
        return self._payloads[source]


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
        assert sorted(handle) == ["metadata", "rank0", "rank2"]
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
    """A write failure must not leave the output file open."""
    comm = FakeComm(rank=0, size=2, payloads={1: payload(NS)})
    configured._comm = comm
    configured._rank, configured._size = 0, 2

    from scope_profiler import h5writer

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(h5writer.ProfilingWriter, "write_rank", boom)

    with pytest.raises(OSError):
        ProfileManager._collect_payloads(
            payload(NS), write_file=True, need_results=False
        )
    # Reopening proves the handle was released rather than left dangling.
    with h5py.File(configured.file_path, "r") as handle:
        assert "metadata" in handle


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
