"""Opt-in profiling wrappers for ``mpi4py`` communicators and requests.

Importing this module never imports mpi4py. Pass an existing communicator to
``profile_mpi_comm``, or explicitly call ``profile_mpi4py`` to patch predefined
communicators for the duration of a run. Importing scope-profiler therefore
cannot initialize MPI in a serial process.
"""

from __future__ import annotations

import builtins
import pickle
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from scope_profiler.profile_manager import ProfileManager

_COMMUNICATOR_FACTORIES = frozenset(
    {
        "Clone",
        "Create",
        "Create_cart",
        "Create_dist_graph",
        "Create_dist_graph_adjacent",
        "Create_graph",
        "Create_group",
        "Create_intercomm",
        "Dup",
        "Dup_with_info",
        "Idup",
        "Merge",
        "Split",
        "Split_type",
    }
)


@dataclass(frozen=True)
class MPIRegionMetadata:
    """Structured metadata decoded from a Python or native MPI region name."""

    operation: str
    kind: str
    bytes: int
    peer: int
    root: int
    tag: int
    communicator: int
    request: str | None = None


def _kind(operation: str) -> str:
    if operation == "wait":
        return "wait"
    if operation in {"barrier", "bcast", "reduce", "allreduce"}:
        return "collective"
    return "point-to-point"


def format_mpi_region(
    operation: str,
    *,
    bytes: int = -1,
    peer: int = -1,
    root: int = -1,
    tag: int = -1,
    communicator: int = -1,
    request: str | None = None,
) -> str:
    """Return the canonical region name shared by Python and C++ wrappers."""
    name = (
        f"mpi:{operation} kind={_kind(operation)} bytes={bytes} peer={peer} "
        f"root={root} tag={tag} comm={communicator}"
    )
    return f"{name} request={request}" if request is not None else name


def parse_mpi_region(name: str) -> MPIRegionMetadata | None:
    """Decode a canonical MPI region name, or return ``None`` for another region."""
    if not name.startswith("mpi:"):
        return None
    parts = name.split()
    try:
        values = dict(part.split("=", 1) for part in parts[1:])
        return MPIRegionMetadata(
            operation=parts[0][4:],
            kind=values["kind"],
            bytes=int(values["bytes"]),
            peer=int(values["peer"]),
            root=int(values["root"]),
            tag=int(values["tag"]),
            communicator=int(values["comm"]),
            request=values.get("request"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _tags(
    operation: str,
    bytes: int,
    peer: int,
    root: int,
    tag: int,
    communicator: int,
    request: str | None = None,
) -> tuple[str, ...]:
    values = (
        "mpi",
        f"mpi.kind:{_kind(operation)}",
        f"mpi.operation:{operation}",
        f"mpi.bytes:{bytes}",
        f"mpi.peer:{peer}",
        f"mpi.root:{root}",
        f"mpi.tag:{tag}",
        f"mpi.communicator:{communicator}",
    )
    return values + ((f"mpi.request:{request}",) if request is not None else ())


def _communicator_id(communicator: Any) -> int:
    try:
        return int(communicator.py2f())
    except (AttributeError, TypeError, ValueError):
        return id(communicator)


def _datatype_size(datatype: Any) -> int | None:
    get_size = getattr(datatype, "Get_size", None)
    if get_size is None:
        return None
    try:
        return int(get_size())
    except (TypeError, ValueError):
        return None


def message_nbytes(value: Any, *, pickle_objects: bool = True) -> int:
    """Estimate payload bytes for an mpi4py object or buffer specification.

    Buffer-mode calls are exact when their datatype exposes ``Get_size``.
    Object-mode values use their buffer size when possible. With
    ``pickle_objects=True`` (the default for this explicit helper), other
    objects use the size of their protocol-5 pickle. ``-1`` means unknown.
    """
    if value is None:
        return -1

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        datatype = value[-1]
        datatype_size = _datatype_size(datatype)
        if hasattr(datatype, "Get_size") and datatype_size is None:
            return -1
        if datatype_size is not None:
            if len(value) >= 3 and isinstance(value[-2], int):
                return int(value[-2]) * datatype_size
            try:
                return memoryview(value[0]).nbytes
            except (TypeError, ValueError):
                return -1

    try:
        return memoryview(value).nbytes
    except (TypeError, ValueError):
        if not pickle_objects:
            return -1
        try:
            return len(pickle.dumps(value, protocol=5))
        except (AttributeError, pickle.PickleError, TypeError):
            return -1


class ProfiledMPIRequest:
    """A request proxy that attributes blocking completion time to MPI wait."""

    def __init__(
        self,
        request: Any,
        *,
        operation: str,
        bytes: int,
        peer: int,
        tag: int,
        communicator: int,
    ) -> None:
        self._request = request
        self._operation = operation
        self._bytes = bytes
        self._peer = peer
        self._tag = tag
        self._communicator = communicator

    def _wait_region(self):
        name = format_mpi_region(
            "wait",
            bytes=self._bytes,
            peer=self._peer,
            tag=self._tag,
            communicator=self._communicator,
            request=self._operation,
        )
        return ProfileManager.profile_region(
            name,
            tags=_tags(
                "wait",
                self._bytes,
                self._peer,
                -1,
                self._tag,
                self._communicator,
                self._operation,
            ),
        )

    def wait(self, status=None):
        with self._wait_region():
            return self._request.wait(status)

    def Wait(self, status=None):
        with self._wait_region():
            return self._request.Wait(status)

    def __getattr__(self, name: str):
        return getattr(self._request, name)


class ProfiledMPIComm:
    """A transparent communicator proxy for common mpi4py operations."""

    def __init__(self, communicator: Any, *, pickle_object_sizes: bool = False) -> None:
        self._communicator = communicator
        self._communicator_id = _communicator_id(communicator)
        self._pickle_object_sizes = pickle_object_sizes

    def _nbytes(self, value: Any) -> int:
        return message_nbytes(value, pickle_objects=self._pickle_object_sizes)

    @property
    def communicator(self) -> Any:
        """The unwrapped mpi4py communicator."""
        return self._communicator

    def _region(
        self,
        operation: str,
        *,
        bytes: int = -1,
        peer: int = -1,
        root: int = -1,
        tag: int = -1,
    ):
        name = format_mpi_region(
            operation,
            bytes=bytes,
            peer=peer,
            root=root,
            tag=tag,
            communicator=self._communicator_id,
        )
        return ProfileManager.profile_region(
            name,
            tags=_tags(
                operation,
                bytes,
                peer,
                root,
                tag,
                self._communicator_id,
            ),
        )

    def _request(self, request, operation, bytes, peer, tag):
        return ProfiledMPIRequest(
            request,
            operation=operation,
            bytes=bytes,
            peer=peer,
            tag=tag,
            communicator=self._communicator_id,
        )

    def send(self, obj, dest: int, tag: int = 0):
        size = self._nbytes(obj)
        with self._region("send", bytes=size, peer=dest, tag=tag):
            return self._communicator.send(obj, dest=dest, tag=tag)

    def recv(self, buf=None, source: int = -1, tag: int = -1, status=None):
        with self._region("recv", bytes=self._nbytes(buf), peer=source, tag=tag):
            return self._communicator.recv(
                buf=buf, source=source, tag=tag, status=status
            )

    def isend(self, obj, dest: int, tag: int = 0) -> ProfiledMPIRequest:
        size = self._nbytes(obj)
        with self._region("isend", bytes=size, peer=dest, tag=tag):
            request = self._communicator.isend(obj, dest=dest, tag=tag)
        return self._request(request, "isend", size, dest, tag)

    def irecv(self, buf=None, source: int = -1, tag: int = -1) -> ProfiledMPIRequest:
        size = self._nbytes(buf)
        with self._region("irecv", bytes=size, peer=source, tag=tag):
            request = self._communicator.irecv(buf=buf, source=source, tag=tag)
        return self._request(request, "irecv", size, source, tag)

    def Send(self, buf, dest: int, tag: int = 0):
        size = self._nbytes(buf)
        with self._region("send", bytes=size, peer=dest, tag=tag):
            return self._communicator.Send(buf, dest=dest, tag=tag)

    def Recv(self, buf, source: int = -1, tag: int = -1, status=None):
        size = self._nbytes(buf)
        with self._region("recv", bytes=size, peer=source, tag=tag):
            return self._communicator.Recv(buf, source=source, tag=tag, status=status)

    def Isend(self, buf, dest: int, tag: int = 0) -> ProfiledMPIRequest:
        size = self._nbytes(buf)
        with self._region("isend", bytes=size, peer=dest, tag=tag):
            request = self._communicator.Isend(buf, dest=dest, tag=tag)
        return self._request(request, "isend", size, dest, tag)

    def Irecv(self, buf, source: int = -1, tag: int = -1) -> ProfiledMPIRequest:
        size = self._nbytes(buf)
        with self._region("irecv", bytes=size, peer=source, tag=tag):
            request = self._communicator.Irecv(buf, source=source, tag=tag)
        return self._request(request, "irecv", size, source, tag)

    def barrier(self):
        with self._region("barrier", bytes=0):
            return self._communicator.barrier()

    def Barrier(self):
        with self._region("barrier", bytes=0):
            return self._communicator.Barrier()

    def bcast(self, obj, root: int = 0):
        size = self._nbytes(obj)
        with self._region("bcast", bytes=size, root=root):
            return self._communicator.bcast(obj, root=root)

    def Bcast(self, buf, root: int = 0):
        size = self._nbytes(buf)
        with self._region("bcast", bytes=size, root=root):
            return self._communicator.Bcast(buf, root=root)

    def allreduce(self, sendobj, op=None):
        size = self._nbytes(sendobj)
        with self._region("allreduce", bytes=size):
            if op is None:
                return self._communicator.allreduce(sendobj)
            return self._communicator.allreduce(sendobj, op=op)

    def Allreduce(self, sendbuf, recvbuf, op=None):
        size = self._nbytes(sendbuf)
        with self._region("allreduce", bytes=size):
            if op is None:
                return self._communicator.Allreduce(sendbuf, recvbuf)
            return self._communicator.Allreduce(sendbuf, recvbuf, op=op)

    def reduce(self, sendobj, op=None, root: int = 0):
        size = self._nbytes(sendobj)
        with self._region("reduce", bytes=size, root=root):
            if op is None:
                return self._communicator.reduce(sendobj, root=root)
            return self._communicator.reduce(sendobj, op=op, root=root)

    def Reduce(self, sendbuf, recvbuf, op=None, root: int = 0):
        size = self._nbytes(sendbuf)
        with self._region("reduce", bytes=size, root=root):
            if op is None:
                return self._communicator.Reduce(sendbuf, recvbuf, root=root)
            return self._communicator.Reduce(sendbuf, recvbuf, op=op, root=root)

    def __getattr__(self, name: str):
        attribute = getattr(self._communicator, name)
        if name not in _COMMUNICATOR_FACTORIES or not callable(attribute):
            return attribute

        def create_profiled_communicator(*args, **kwargs):
            created = attribute(*args, **kwargs)
            if name == "Idup":
                communicator, request = created
                return (
                    profile_mpi_comm(
                        communicator,
                        pickle_object_sizes=self._pickle_object_sizes,
                    ),
                    request,
                )
            return profile_mpi_comm(
                created,
                pickle_object_sizes=self._pickle_object_sizes,
            )

        return create_profiled_communicator


def profile_mpi_comm(
    communicator: Any, *, pickle_object_sizes: bool = False
) -> ProfiledMPIComm:
    """Wrap a communicator without importing mpi4py.

    Set ``pickle_object_sizes=True`` to measure serialized object-mode payloads
    at the cost of one extra serialization per call. It is off by default to
    avoid overhead and repeated ``__reduce__`` side effects.
    """
    if isinstance(communicator, ProfiledMPIComm):
        return communicator
    return ProfiledMPIComm(communicator, pickle_object_sizes=pickle_object_sizes)


@contextmanager
def profile_mpi4py(
    mpi_module: Any = None,
    *,
    pickle_object_sizes: bool = False,
    lazy: bool = False,
) -> Iterator[None]:
    """Temporarily profile mpi4py's predefined communicators.

    When *lazy* is true, mpi4py is wrapped only if the target imports it. This
    lets ``scope-profiler run`` enable MPI-call profiling by default without
    importing or initializing MPI for an ordinary serial program. Both globals
    and Python's import function are restored even when the program raises.
    """
    originals = []
    patched_modules = set()

    def wrap_predefined(module):
        module_id = id(module)
        if module_id in patched_modules:
            return
        patched_modules.add(module_id)
        for name in ("COMM_WORLD", "COMM_SELF"):
            communicator = getattr(module, name, None)
            if communicator is not None:
                originals.append((module, name, communicator))
                setattr(
                    module,
                    name,
                    profile_mpi_comm(
                        communicator,
                        pickle_object_sizes=pickle_object_sizes,
                    ),
                )

    original_import = None
    if mpi_module is not None:
        wrap_predefined(mpi_module)
    elif lazy:
        loaded_mpi = sys.modules.get("mpi4py.MPI")
        if loaded_mpi is not None:
            wrap_predefined(loaded_mpi)

        original_import = builtins.__import__

        def import_and_profile(name, *args, **kwargs):
            imported = original_import(name, *args, **kwargs)
            loaded_mpi = sys.modules.get("mpi4py.MPI")
            if loaded_mpi is not None:
                wrap_predefined(loaded_mpi)
            return imported

        builtins.__import__ = import_and_profile
    else:
        try:
            from mpi4py import MPI as mpi_module
        except ImportError as error:
            raise RuntimeError(
                "--mpi-calls requires mpi4py; install scope-profiler[mpi]"
            ) from error

        wrap_predefined(mpi_module)
    try:
        yield
    finally:
        if original_import is not None:
            builtins.__import__ = original_import
        for module, name, communicator in reversed(originals):
            setattr(module, name, communicator)


__all__ = [
    "MPIRegionMetadata",
    "ProfiledMPIComm",
    "ProfiledMPIRequest",
    "format_mpi_region",
    "message_nbytes",
    "parse_mpi_region",
    "profile_mpi4py",
    "profile_mpi_comm",
]
