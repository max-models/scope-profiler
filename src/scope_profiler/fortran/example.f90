!> Runnable example of the Fortran region API.
!!
!!     gfortran -cpp -O2 -c scope_profiler.F90
!!     gfortran -cpp -O2 example.f90 scope_profiler.o -o example
!!     ./example
!!     scope-profiler import-fortran . -o profiling_data.h5
!!     scope-profiler inspect profiling_data.h5
program example
   use scope_profiler
   use, intrinsic :: iso_fortran_env, only: int64, output_unit
   implicit none

   integer :: step, solve, assemble, io
   real(kind=8) :: total

   call sp_init("profile")

   ! Resolve the names once; the handles are what the hot loop uses.
   solve = sp_region("solve")
   assemble = sp_region("assemble")
   io = sp_region("checkpoint")

   do step = 1, 20
      call sp_begin(assemble)
      total = busy_work(20000)
      call sp_end(assemble)

      call sp_begin(solve)
      total = total + busy_work(50000)
      call sp_end(solve)

      if (mod(step, 10) == 0) then
         call sp_begin(io)
         total = total + busy_work(5000)
         call sp_end(io)
      end if
   end do

   ! The convenience form, and a recursive region.
   call sp_begin_name("fibonacci")
   step = fib(12)
   call sp_end_name("fibonacci")

   call sp_finalize()

   write (output_unit, "(a,i0,a)") "solve entered ", sp_num_calls(solve), " time(s)"
   write (output_unit, "(a,f12.4)") "checksum (ignore): ", total + real(step, kind=8)
   write (output_unit, "(a)") "wrote profile_rank00000.spt"

contains

   !> Something the compiler cannot optimise away.
   function busy_work(n) result(acc)
      integer, intent(in) :: n
      real(kind=8) :: acc
      integer :: i

      acc = 0.0d0
      do i = 1, n
         acc = acc + sqrt(real(i, kind=8))
      end do
   end function busy_work

   !> Recursion: every invocation reserves its own slot in the same region.
   recursive function fib(n) result(value)
      integer, intent(in) :: n
      integer :: value
      integer :: id

      id = sp_region("fib_call")
      call sp_begin(id)
      if (n < 2) then
         value = n
      else
         value = fib(n - 1) + fib(n - 2)
      end if
      call sp_end(id)
   end function fib

end program example
