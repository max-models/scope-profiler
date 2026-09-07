"""Profile exact-size NumPy buffer operations with mpi4py.

Run with at least two ranks:

    mpirun -n 2 scope-profiler run -q \
        -o mpi_numpy_calls.h5 examples/ex_mpi_numpy_wrappers.py
"""

import numpy as np
from mpi4py import MPI


def main() -> None:
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        raise RuntimeError("this example requires at least two MPI ranks")

    if comm.rank == 0:
        values = np.arange(8, dtype=np.float64)
        comm.Isend([values, MPI.DOUBLE], dest=1, tag=17).Wait()
    elif comm.rank == 1:
        received = np.empty(8, dtype=np.float64)
        comm.Irecv([received, MPI.DOUBLE], source=0, tag=17).Wait()
        np.testing.assert_array_equal(received, np.arange(8, dtype=np.float64))

    broadcast_value = np.array([42 if comm.rank == 0 else 0], dtype=np.int32)
    comm.Bcast([broadcast_value, MPI.INT], root=0)
    assert broadcast_value[0] == 42

    local = np.array([comm.rank + 1], dtype=np.int32)
    total = np.zeros(1, dtype=np.int32)
    comm.Allreduce([local, MPI.INT], [total, MPI.INT], op=MPI.SUM)
    assert total[0] == comm.size * (comm.size + 1) // 2

    comm.Barrier()


if __name__ == "__main__":
    main()
