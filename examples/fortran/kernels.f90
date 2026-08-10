!> A small Fortran solver library that profiles its own internals.
!!
!! This is the shape a real kernel library takes: the caller knows nothing
!! about the regions inside it, and does not have to. `standalone.f90` drives
!! it from Fortran; `run_mixed.py` drives the very same code from Python
!! through f2py. Either way the regions below end up in the profile.
!!
!! Region names are prefixed `fortran:` so they cannot collide with the
!! driver's own regions -- scope-profiler refuses to merge a name recorded on
!! both sides, since that would double-count a wrapper and the region inside it.
!!
!! One f2py trap is worth knowing, and is why this file says `double precision`
!! rather than the more modern `real(dp)`: f2py does not resolve *named* kind
!! parameters. Wrap `real(dp)` with `dp = kind(1.0d0)` -- or `real(real64)`
!! from `iso_fortran_env` -- and the extension builds, runs, records its
!! regions correctly, and hands back garbage for the value. Spell kinds
!! literally (`double precision`, or `real(kind=8)`) in anything f2py wraps.
module kernels
   use scope_profiler
   implicit none
   private

   public :: start_profiling, stop_profiling, jacobi_solve, checkpoint

contains

   !> Begin recording. Call once, before any kernel.
   !!
   !! @param prefix  output prefix; the trace lands in `<prefix>_rank<NNNNN>.spt`
   !! @param rank    MPI rank, so each process writes its own trace
   subroutine start_profiling(prefix, rank)
      character(len=*), intent(in) :: prefix
      integer, intent(in) :: rank

      call sp_init(prefix, rank=rank)
   end subroutine start_profiling

   !> Stop recording and write the trace. Call before the driver finalizes.
   subroutine stop_profiling()
      call sp_finalize()
   end subroutine stop_profiling

   !> A Jacobi smoother on a 1-D grid, profiled sweep by sweep.
   !!
   !! Two regions are recorded per call: the stencil update, and the residual
   !! reduction that decides whether to stop.
   !!
   !! @param n           grid points
   !! @param iterations  sweeps to perform
   !! @param residual    final residual, returned so nothing is optimised away
   subroutine jacobi_solve(n, iterations, residual)
      integer, intent(in) :: n
      integer, intent(in) :: iterations
      double precision, intent(out) :: residual

      double precision, allocatable :: u(:), u_new(:)
      integer :: sweep, i, stencil_id, residual_id

      stencil_id = sp_region("fortran:stencil")
      residual_id = sp_region("fortran:residual")

      allocate (u(n), u_new(n))
      u = 0.0d0
      u(1) = 1.0d0
      u(n) = 0.0d0
      u_new = u

      residual = 0.0d0
      do sweep = 1, iterations
         call sp_begin(stencil_id)
         do i = 2, n - 1
            u_new(i) = 0.5d0*(u(i - 1) + u(i + 1))
         end do
         u_new(1) = u(1)
         u_new(n) = u(n)
         call sp_end(stencil_id)

         call sp_begin(residual_id)
         residual = 0.0d0
         do i = 2, n - 1
            residual = residual + abs(u_new(i) - u(i))
         end do
         u = u_new
         call sp_end(residual_id)
      end do

      deallocate (u, u_new)
   end subroutine jacobi_solve

   !> Stand-in for writing a checkpoint: a region that is entered rarely.
   !!
   !! Rare regions are worth marking too -- they are what a Gantt chart makes
   !! obvious and a total-time table hides.
   subroutine checkpoint(n)
      integer, intent(in) :: n
      integer :: id, i
      double precision :: acc

      id = sp_region("fortran:checkpoint")
      call sp_begin(id)
      acc = 0.0d0
      do i = 1, n
         acc = acc + sqrt(real(i, kind=8))
      end do
      if (acc < 0.0d0) print *, acc   ! never taken; defeats the optimiser
      call sp_end(id)
   end subroutine checkpoint

end module kernels
