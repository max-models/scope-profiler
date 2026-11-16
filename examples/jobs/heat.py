"""Toy MPI workload for the profiling-jobs example: a 1D heat equation solved with Jacobi sweeps.

Run directly (``mpirun -n 4 python heat.py --cells 100000 --steps 200``) or
through ``scope-profiler jobs run case.toml``, which passes ``--profile-out``
with the run's ``{profile}`` path.
"""

import argparse

import numpy as np

import scope_profiler as sp

try:
    from mpi4py import MPI
except ImportError:  # serial fallback
    MPI = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--profile-out", default="profiling_data.h5")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD if MPI is not None else None
    rank = comm.Get_rank() if comm else 0
    size = comm.Get_size() if comm else 1

    with sp.session(file_path=args.profile_out):
        local = args.cells // size + (rank < args.cells % size)
        u = np.zeros(local + 2)
        u[1:-1] = np.sin(np.linspace(0, np.pi, local))
        for _ in range(args.steps):
            with sp.region("heat.halo_exchange"):
                if comm is not None and size > 1:
                    left, right = rank - 1, rank + 1
                    if left >= 0:
                        u[0] = comm.sendrecv(u[1], dest=left, source=left)
                    if right < size:
                        u[-1] = comm.sendrecv(u[-2], dest=right, source=right)
            with sp.region("heat.update"):
                u[1:-1] = 0.5 * u[1:-1] + 0.25 * (u[:-2] + u[2:])
    if rank == 0:
        print(f"done: {args.cells} cells, {args.steps} steps on {size} rank(s)")


if __name__ == "__main__":
    main()
