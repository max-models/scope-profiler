"""MPI wrapper behavior without requiring an MPI runtime or launcher."""

import builtins
import pickle
import sys
from types import SimpleNamespace

import pytest

from scope_profiler import ProfileManager
from scope_profiler.mpi_wrappers import (
    ProfiledMPIComm,
    ProfiledMPIRequest,
    format_mpi_region,
    message_nbytes,
    parse_mpi_region,
    profile_mpi_comm,
    profile_mpi4py,
)


class FakeDatatype:
    def __init__(self, size=4):
        self.size = size

    def Get_size(self):
        if self.size is None:
            raise ValueError("unknown datatype")
        return self.size


class FakeRequest:
    marker = "underlying request"

    def __init__(self, operation):
        self.operation = operation

    def wait(self, status=None):
        return ("wait", self.operation, status)

    def Wait(self, status=None):
        return ("Wait", self.operation, status)


class FakeComm:
    passthrough = "communicator attribute"

    def __init__(self, handle=7):
        self.handle = handle
        self.calls = []

    def py2f(self):
        return self.handle

    def _call(self, operation, *args, **kwargs):
        self.calls.append((operation, args, kwargs))
        return operation

    def send(self, *args, **kwargs):
        return self._call("send", *args, **kwargs)

    def recv(self, *args, **kwargs):
        return self._call("recv", *args, **kwargs)

    def isend(self, *args, **kwargs):
        self._call("isend", *args, **kwargs)
        return FakeRequest("isend")

    def irecv(self, *args, **kwargs):
        self._call("irecv", *args, **kwargs)
        return FakeRequest("irecv")

    def Send(self, *args, **kwargs):
        return self._call("Send", *args, **kwargs)

    def Recv(self, *args, **kwargs):
        return self._call("Recv", *args, **kwargs)

    def Isend(self, *args, **kwargs):
        self._call("Isend", *args, **kwargs)
        return FakeRequest("Isend")

    def Irecv(self, *args, **kwargs):
        self._call("Irecv", *args, **kwargs)
        return FakeRequest("Irecv")

    def barrier(self):
        return self._call("barrier")

    def Barrier(self):
        return self._call("Barrier")

    def bcast(self, *args, **kwargs):
        return self._call("bcast", *args, **kwargs)

    def Bcast(self, *args, **kwargs):
        return self._call("Bcast", *args, **kwargs)

    def allreduce(self, *args, **kwargs):
        return self._call("allreduce", *args, **kwargs)

    def Allreduce(self, *args, **kwargs):
        return self._call("Allreduce", *args, **kwargs)

    def reduce(self, *args, **kwargs):
        return self._call("reduce", *args, **kwargs)

    def Reduce(self, *args, **kwargs):
        return self._call("Reduce", *args, **kwargs)

    def Split(self, *args, **kwargs):
        self._call("Split", *args, **kwargs)
        return FakeComm(handle=8)

    def Idup(self, *args, **kwargs):
        self._call("Idup", *args, **kwargs)
        return FakeComm(handle=9), FakeRequest("Idup")


def test_region_name_has_stable_cross_language_metadata_order():
    name = format_mpi_region("send", bytes=32, peer=2, tag=9, communicator=7)
    assert name == ("mpi:send kind=point-to-point bytes=32 peer=2 root=-1 tag=9 comm=7")
    metadata = parse_mpi_region(name)
    assert metadata is not None
    assert metadata.operation == "send"
    assert metadata.bytes == 32
    assert metadata.peer == 2
    assert metadata.communicator == 7
    assert metadata.request is None
    assert format_mpi_region(
        "wait", bytes=32, peer=2, tag=9, communicator=7, request="isend"
    ).endswith(" request=isend")
    assert parse_mpi_region("ordinary-region") is None
    assert parse_mpi_region("mpi:send kind=broken") is None


def test_message_nbytes_handles_buffers_specs_objects_and_unknowns():
    assert message_nbytes(bytearray(11)) == 11
    assert message_nbytes([bytearray(20), 3, FakeDatatype(4)]) == 12
    assert message_nbytes([bytearray(20), FakeDatatype(4)]) == 20
    assert message_nbytes([object(), FakeDatatype(4)]) == -1
    assert message_nbytes([object(), FakeDatatype(None)]) == -1
    assert message_nbytes([1, 2]) == len(pickle.dumps([1, 2], protocol=5))
    assert message_nbytes(None) == -1
    value = {"answer": 42}
    assert message_nbytes(value) == len(pickle.dumps(value, protocol=5))
    assert message_nbytes(value, pickle_objects=False) == -1
    assert message_nbytes(lambda: None) == -1


def test_python_proxy_profiles_object_buffer_collective_and_wait_calls():
    raw = FakeComm()
    communicator = profile_mpi_comm(raw)
    assert isinstance(communicator, ProfiledMPIComm)
    assert profile_mpi_comm(communicator) is communicator
    assert communicator.communicator is raw
    assert communicator.passthrough == "communicator attribute"

    buffer = [bytearray(16), 4, FakeDatatype(4)]
    with ProfileManager.session(
        deactivate_file_output=True, return_results=True, verbose=False
    ) as run:
        assert communicator.send(b"abc", 2, 3) == "send"
        assert communicator.recv(bytearray(8), 1, 4, "status") == "recv"
        request = communicator.isend(b"abcd", 3, 5)
        assert isinstance(request, ProfiledMPIRequest)
        assert request.marker == "underlying request"
        assert request.wait("lower-status") == ("wait", "isend", "lower-status")
        request = communicator.irecv(bytearray(5), 4, 6)
        assert request.Wait("upper-status") == ("Wait", "irecv", "upper-status")

        assert communicator.Send(buffer, 2, 7) == "Send"
        assert communicator.Recv(buffer, 1, 8, "status") == "Recv"
        request = communicator.Isend(buffer, 3, 9)
        assert request.Wait() == ("Wait", "Isend", None)
        request = communicator.Irecv(buffer, 4, 10)
        assert request.wait() == ("wait", "Irecv", None)

        assert communicator.barrier() == "barrier"
        assert communicator.Barrier() == "Barrier"
        assert communicator.bcast(b"abc", root=1) == "bcast"
        assert communicator.Bcast(buffer, root=2) == "Bcast"
        assert communicator.allreduce(3) == "allreduce"
        assert communicator.allreduce(3, op="sum") == "allreduce"
        assert communicator.Allreduce(buffer, bytearray(16)) == "Allreduce"
        assert communicator.Allreduce(buffer, bytearray(16), op="sum") == "Allreduce"
        assert communicator.reduce(3, root=1) == "reduce"
        assert communicator.reduce(3, op="sum", root=2) == "reduce"
        assert communicator.Reduce(buffer, bytearray(16), root=1) == "Reduce"
        assert communicator.Reduce(buffer, bytearray(16), op="sum", root=2) == "Reduce"

    results = run.results
    assert results is not None
    operations = {
        name.split()[0].removeprefix("mpi:")
        for name in results.region_names
        if name.startswith("mpi:")
    }
    assert operations == {
        "send",
        "recv",
        "isend",
        "irecv",
        "wait",
        "barrier",
        "bcast",
        "allreduce",
        "reduce",
    }
    send = next(region for region in results if region.name.startswith("mpi:send"))
    assert "mpi.operation:send" in send.tags
    assert "mpi.peer:2" in send.tags
    assert "mpi.communicator:7" in send.tags


def test_proxy_fallback_communicator_id_and_exception_timing():
    class NoHandleComm(FakeComm):
        def py2f(self):
            raise TypeError("no Fortran handle")

        def send(self, *args, **kwargs):
            raise RuntimeError("MPI failure")

    raw = NoHandleComm()
    communicator = ProfiledMPIComm(raw)
    with ProfileManager.session(
        deactivate_file_output=True, return_results=True, verbose=False
    ) as run:
        with pytest.raises(RuntimeError, match="MPI failure"):
            communicator.send(b"x", 1)

    assert run.results is not None
    mpi_names = [name for name in run.results.region_names if name.startswith("mpi:")]
    assert len(mpi_names) == 1
    assert f"comm={id(raw)}" in mpi_names[0]


def test_object_size_estimation_is_explicitly_opt_in():
    raw = FakeComm(handle=11)
    value = {"answer": 42}
    communicator = profile_mpi_comm(raw, pickle_object_sizes=True)

    with ProfileManager.session(
        deactivate_file_output=True, return_results=True, verbose=False
    ) as run:
        communicator.send(value, 2)

    expected = len(pickle.dumps(value, protocol=5))
    mpi_names = [name for name in run.results.region_names if name.startswith("mpi:")]
    assert len(mpi_names) == 1
    assert f"bytes={expected}" in mpi_names[0]


def test_profile_mpi4py_temporarily_wraps_predefined_communicators():
    class FakeMPI:
        COMM_WORLD = FakeComm(handle=21)
        COMM_SELF = FakeComm(handle=22)

    world = FakeMPI.COMM_WORLD
    self_comm = FakeMPI.COMM_SELF
    with profile_mpi4py(FakeMPI):
        assert isinstance(FakeMPI.COMM_WORLD, ProfiledMPIComm)
        assert isinstance(FakeMPI.COMM_SELF, ProfiledMPIComm)
        assert FakeMPI.COMM_WORLD.communicator is world
        assert FakeMPI.COMM_SELF.communicator is self_comm

    assert FakeMPI.COMM_WORLD is world
    assert FakeMPI.COMM_SELF is self_comm


def test_proxy_propagates_through_communicator_constructors():
    communicator = profile_mpi_comm(FakeComm())

    split = communicator.Split(1, 2)
    assert isinstance(split, ProfiledMPIComm)
    assert split.communicator.handle == 8

    duplicate, request = communicator.Idup()
    assert isinstance(duplicate, ProfiledMPIComm)
    assert duplicate.communicator.handle == 9
    assert request.operation == "Idup"


def test_profile_mpi4py_restores_communicators_after_an_error():
    class FakeMPI:
        COMM_WORLD = FakeComm()

    world = FakeMPI.COMM_WORLD
    with pytest.raises(RuntimeError, match="target failed"):
        with profile_mpi4py(FakeMPI):
            raise RuntimeError("target failed")

    assert FakeMPI.COMM_WORLD is world


def test_profile_mpi4py_explains_when_mpi4py_is_not_installed(monkeypatch):
    real_import = builtins.__import__

    def import_without_mpi4py(name, *args, **kwargs):
        if name == "mpi4py":
            raise ImportError("mpi4py unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_mpi4py)
    with pytest.raises(RuntimeError, match=r"scope-profiler\[mpi\]"):
        with profile_mpi4py():
            pass


def test_profile_mpi4py_lazy_mode_waits_for_the_target_import(monkeypatch):
    class FakeMPI:
        COMM_WORLD = FakeComm(handle=31)
        COMM_SELF = FakeComm(handle=32)

    imports = []

    def target_import(name, *args, **kwargs):
        imports.append(name)
        if name == "mpi4py":
            monkeypatch.setitem(sys.modules, "mpi4py.MPI", FakeMPI)
            return SimpleNamespace(MPI=FakeMPI)
        raise ImportError(name)

    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    monkeypatch.setattr(builtins, "__import__", target_import)
    world = FakeMPI.COMM_WORLD
    with profile_mpi4py(lazy=True):
        assert imports == []
        package = builtins.__import__("mpi4py", fromlist=("MPI",))
        assert package.MPI is FakeMPI
        assert isinstance(FakeMPI.COMM_WORLD, ProfiledMPIComm)
        builtins.__import__("mpi4py", fromlist=("MPI",))

    assert FakeMPI.COMM_WORLD is world
    assert builtins.__import__ is target_import


def test_profile_mpi4py_can_eagerly_import_mpi4py(monkeypatch):
    class FakeMPI:
        COMM_WORLD = FakeComm(handle=35)

    world = FakeMPI.COMM_WORLD

    def target_import(name, *args, **kwargs):
        if name == "mpi4py":
            return SimpleNamespace(MPI=FakeMPI)
        raise ImportError(name)

    monkeypatch.setattr(builtins, "__import__", target_import)
    with profile_mpi4py():
        assert isinstance(FakeMPI.COMM_WORLD, ProfiledMPIComm)

    assert FakeMPI.COMM_WORLD is world


def test_profile_mpi4py_lazy_mode_wraps_an_already_loaded_module(monkeypatch):
    class FakeMPI:
        COMM_WORLD = FakeComm(handle=41)

    monkeypatch.setitem(sys.modules, "mpi4py.MPI", FakeMPI)
    world = FakeMPI.COMM_WORLD
    with profile_mpi4py(lazy=True):
        assert isinstance(FakeMPI.COMM_WORLD, ProfiledMPIComm)

    assert FakeMPI.COMM_WORLD is world
