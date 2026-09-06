"""Profile mpi4py communication operations and their message metadata.

Run with at least two ranks::

    mpirun -n 2 python examples/ex_mpi_wrappers.py
"""

from mpi4py import MPI

from scope_profiler import ProfileManager, profile_mpi_comm


def main() -> None:
    comm = profile_mpi_comm(MPI.COMM_WORLD)

    with ProfileManager.session(file_path="mpi_calls.h5", verbose=False):
        if comm.rank == 0:
            request = comm.isend({"work": 42}, dest=1, tag=7)
            request.wait()
        elif comm.rank == 1:
            comm.recv(source=0, tag=7)

        comm.allreduce(comm.rank)
        comm.barrier()


if __name__ == "__main__":
    main()
