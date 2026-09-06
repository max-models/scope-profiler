#include <scope_profiler_mpi.hpp>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    sp_init("cmake-mpi-profile", rank);

    int value = rank;
    if (rank == 0) {
        sp::mpi::Request request = sp::mpi::isend(
            &value, 1, MPI_INT, 1, 7, MPI_COMM_WORLD);
        sp::mpi::wait(request);
    } else if (rank == 1) {
        sp::mpi::recv(&value, 1, MPI_INT, 0, 7, MPI_COMM_WORLD);
    }

    int total = 0;
    sp::mpi::allreduce(
        &rank, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    sp::mpi::barrier(MPI_COMM_WORLD);

    int profiler_status = sp_finalize();
    MPI_Finalize();
    return profiler_status;
}
