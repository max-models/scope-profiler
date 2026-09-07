"""Enable automatic mpi4py instrumentation through ProfilingOptions.

Run with at least two ranks:

    mpirun -n 2 python examples/ex_mpi_profiling_options.py
"""

import numpy as np
from mpi4py import MPI

from scope_profiler import ProfileManager, ProfilingOptions


def main() -> None:
    options = ProfilingOptions(
        profile_mpi_calls=True,
        file_path="mpi_profiling_options.h5",
    )

    with ProfileManager.session(options=options, verbose=True):
        # Obtain the communicator after entering the session so this reference
        # receives the automatic profiling proxy.
        comm = MPI.COMM_WORLD
        if comm.size < 2:
            raise RuntimeError("this example requires at least two MPI ranks")

        value = np.array([comm.rank], dtype=np.int32)
        if comm.rank == 0:
            comm.Send([value, MPI.INT], dest=1, tag=23)
        elif comm.rank == 1:
            comm.Recv([value, MPI.INT], source=0, tag=23)
            assert value[0] == 0

        # Communicators derived from a profiled communicator remain profiled.
        local_comm = comm.Split(color=comm.rank % 2, key=comm.rank)
        local_total = local_comm.allreduce(comm.rank, op=MPI.SUM)
        assert local_total >= 0
        local_comm.Free()

        comm.Barrier()

    if MPI.COMM_WORLD.rank == 0:
        print("wrote mpi_profiling_options.h5")


if __name__ == "__main__":
    main()
