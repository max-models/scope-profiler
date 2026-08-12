!> A pure Fortran program, profiled with no Python at run time.
!!
!!     make standalone && make run-standalone
!!
!! Marks its own regions around the kernels' regions, so the resulting profile
!! shows both levels. Takes an optional rank argument, which is how you would
!! pass `MPI_Comm_rank` under a launcher:
!!
!!     mpirun -n 4 ./standalone   # if you build it with mpif90; see README.md
program standalone
   use scope_profiler
   use kernels
   use, intrinsic :: iso_fortran_env, only: real64, output_unit
   implicit none

   integer, parameter :: GRID = 20000
   integer, parameter :: STEPS = 20
   integer, parameter :: SWEEPS = 5

   integer :: step, rank, timestep_id, setup_id
   character(len=32) :: argument
   real(real64) :: residual

   ! Optional rank argument, so several processes can write distinct traces.
   rank = 0
   if (command_argument_count() >= 1) then
      call get_command_argument(1, argument)
      read (argument, *) rank
   end if

   call start_profiling("standalone", rank)

   setup_id = sp_region("fortran:setup")
   timestep_id = sp_region("fortran:timestep")

   call sp_begin(setup_id)
   call checkpoint(50000)          ! pretend: read input, build the grid
   call sp_end(setup_id)

   do step = 1, STEPS
      call sp_begin(timestep_id)
      call jacobi_solve(GRID, SWEEPS, residual)
      call sp_end(timestep_id)

      ! Checkpoint every fifth step: a rare region among frequent ones.
      if (mod(step, 5) == 0) call checkpoint(200000)
   end do

   call stop_profiling()

   write (output_unit, "(a,i0,a)") "rank ", rank, ": done"
   write (output_unit, "(a,es12.5)") "  final residual: ", residual
   write (output_unit, "(a,i5.5,a)") "  wrote standalone_rank", rank, ".spt"
   write (output_unit, "(a)") "  now run: scope-profiler import-fortran . -o profiling_data.h5"

end program standalone
