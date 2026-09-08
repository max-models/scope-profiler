!> Region profiling for Fortran, on the same timeline as scope-profiler.
!!
!! Records nanosecond start/end timestamps for named regions and writes them
!! to a trace file that `scope-profiler import-native` turns into the usual
!! HDF5 output, so a Fortran run gets the same summaries, plots and exports as
!! a Python one.
!!
!! The module is deliberately self-contained: pure Fortran 2008 with
!! `iso_c_binding`, no C source to compile, no HDF5, no MPI. Drop the file into
!! a build and link nothing extra.
!!
!! Usage:
!!
!!     use scope_profiler
!!     integer :: solve
!!
!!     call sp_init("profile", rank=my_rank)   ! rank optional, default 0
!!     solve = sp_region("solve")              ! resolve the name once
!!     do step = 1, nsteps
!!        call sp_begin(solve)
!!        ...
!!        call sp_end(solve)
!!     end do
!!     call sp_finalize()
!!
!! `sp_begin_name("solve")` / `sp_end_name("solve")` exist for convenience but
!! look the name up on every call; prefer the handle form in hot loops.
!!
!! Timestamps come from the same OS clock CPython's `time.perf_counter_ns()`
!! uses -- `CLOCK_MONOTONIC` on Linux, `CLOCK_UPTIME_RAW` on macOS -- so
!! regions recorded here share an epoch with regions recorded by the Python
!! API in the same process tree, and land on one timeline. The right clock is
!! found by probing at run time, so the file is plain Fortran needing no
!! preprocessor and no platform flags.
module scope_profiler
   use, intrinsic :: iso_c_binding, only: c_int, c_long, c_int64_t
   use, intrinsic :: iso_fortran_env, only: int32, int64, error_unit
   implicit none
   private

   public :: sp_init, sp_region, sp_begin, sp_end
   public :: sp_begin_name, sp_end_name, sp_finalize
   public :: sp_num_calls, sp_now_ns, sp_is_active

   !> Longest region name the trace format stores.
   integer, parameter, public :: SP_MAX_NAME = 128

   !> Trace format written by sp_finalize; keep in step with fortran_trace.py.
   character(len=8), parameter :: SP_MAGIC = "SCOPEPRF"
   integer(int32), parameter :: SP_FORMAT_VERSION = 1

   !> Initial slots per region; the buffers double from here as needed.
   integer, parameter :: SP_INITIAL_CAPACITY = 1024

   !> Deepest recursion of a single region that can be open at once.
   integer, parameter :: SP_MAX_DEPTH = 64

   type :: region_t
      character(len=SP_MAX_NAME) :: name = ""
      integer(int64), allocatable :: start_times(:)
      integer(int64), allocatable :: end_times(:)
      integer(int64) :: ptr = 0            !< slots used
      integer(int64) :: capacity = 0
      integer(int64) :: num_calls = 0
      !> Slots reserved by regions still open, innermost last. A recursive
      !! re-entry reserves its own slot instead of overwriting the outer one.
      integer(int64) :: open_slots(SP_MAX_DEPTH) = 0
      integer :: depth = 0
   end type region_t

   type(region_t), allocatable :: regions(:)
   integer :: n_regions = 0
   integer :: rank_id = 0
   character(len=512) :: output_prefix = "scope_profile"
   logical :: active = .false.

   ! clock_gettime(2). timespec is {time_t tv_sec; long tv_nsec}; both are
   ! 64-bit on every 64-bit Unix we target.
   type, bind(c) :: c_timespec
      integer(c_long) :: tv_sec = 0
      integer(c_long) :: tv_nsec = 0
   end type c_timespec

   interface
      function c_clock_gettime(clk_id, tp) bind(c, name="clock_gettime") result(rc)
         import :: c_int, c_timespec
         integer(c_int), value :: clk_id
         type(c_timespec), intent(out) :: tp
         integer(c_int) :: rc
      end function c_clock_gettime
   end interface

   !> Clock ids to try, in order, stopping at the first the OS accepts.
   !!
   !! 1 is CLOCK_MONOTONIC on Linux, which is what CPython's
   !! perf_counter_ns() reads there. macOS rejects id 1 outright (its
   !! CLOCK_MONOTONIC is 6), and that rejection is exactly what identifies the
   !! platform: 8 is CLOCK_UPTIME_RAW, the clock CPython reads on macOS.
   !! Probing beats a preprocessor #ifdef here because gfortran does not
   !! define __APPLE__, so an #ifdef silently picks the wrong branch.
   integer(c_int), parameter :: SP_CLOCK_CANDIDATES(2) = [1_c_int, 8_c_int]

   !> Resolved on first use; -1 means "not yet probed".
   integer(c_int) :: clock_id = -1_c_int

contains

   !> Nanoseconds on the same clock as Python's time.perf_counter_ns().
   !!
   !! Returns a negative value if no monotonic clock could be resolved, which
   !! sp_init() reports and refuses to profile with -- silently handing back 0
   !! would produce a trace full of zero-length regions.
   function sp_now_ns() result(now)
      integer(int64) :: now
      type(c_timespec) :: ts
      integer(c_int) :: rc

      if (clock_id < 0_c_int) call resolve_clock()
      if (clock_id < 0_c_int) then
         now = -1_int64
         return
      end if

      rc = c_clock_gettime(clock_id, ts)
      if (rc /= 0_c_int) then
         now = -1_int64
         return
      end if
      now = int(ts%tv_sec, int64)*1000000000_int64 + int(ts%tv_nsec, int64)
   end function sp_now_ns

   !> Pick the first candidate clock the OS actually supports.
   subroutine resolve_clock()
      type(c_timespec) :: ts
      integer(c_int) :: rc
      integer :: i

      do i = 1, size(SP_CLOCK_CANDIDATES)
         rc = c_clock_gettime(SP_CLOCK_CANDIDATES(i), ts)
         if (rc == 0_c_int) then
            clock_id = SP_CLOCK_CANDIDATES(i)
            return
         end if
      end do
      clock_id = -1_c_int
   end subroutine resolve_clock

   !> Whether sp_init() has been called and regions are being recorded.
   function sp_is_active() result(is_active)
      logical :: is_active
      is_active = active
   end function sp_is_active

   !> Start profiling; must precede any other call.
   !!
   !! @param prefix  output path prefix; sp_finalize writes
   !!                `<prefix>_rank<NNNNN>.spt`
   !! @param rank    MPI rank of this process (default 0). Each rank writes
   !!                its own file; the importer merges them.
   subroutine sp_init(prefix, rank)
      character(len=*), intent(in) :: prefix
      integer, intent(in), optional :: rank

      call resolve_clock()
      if (clock_id < 0_c_int) then
         write (error_unit, "(a)") "scope_profiler: no monotonic clock available "// &
            "(clock_gettime rejected every candidate); profiling is disabled"
         active = .false.
         return
      end if

      if (allocated(regions)) deallocate (regions)
      allocate (regions(16))
      n_regions = 0
      output_prefix = prefix
      rank_id = 0
      if (present(rank)) rank_id = rank
      active = .true.
   end subroutine sp_init

   !> Handle for a region name, creating it on first use.
   !!
   !! Resolve once, outside hot loops, and pass the handle to sp_begin/sp_end.
   function sp_region(name) result(id)
      character(len=*), intent(in) :: name
      integer :: id
      integer :: i
      type(region_t), allocatable :: bigger(:)

      id = 0
      if (.not. active) return

      do i = 1, n_regions
         if (trim(regions(i)%name) == trim(name)) then
            id = i
            return
         end if
      end do

      if (n_regions == size(regions)) then
         allocate (bigger(2*size(regions)))
         bigger(1:n_regions) = regions(1:n_regions)
         call move_alloc(bigger, regions)
      end if

      n_regions = n_regions + 1
      id = n_regions
      regions(id)%name = name
      regions(id)%capacity = SP_INITIAL_CAPACITY
      allocate (regions(id)%start_times(SP_INITIAL_CAPACITY))
      allocate (regions(id)%end_times(SP_INITIAL_CAPACITY))
      regions(id)%ptr = 0
      regions(id)%num_calls = 0
      regions(id)%depth = 0
   end function sp_region

   !> Enter a region. Reserves this call's slot before the work starts, so a
   !! recursive re-entry cannot overwrite it.
   subroutine sp_begin(id)
      integer, intent(in) :: id

      if (.not. active) return
      if (id < 1 .or. id > n_regions) return

      if (regions(id)%ptr >= regions(id)%capacity) call grow(id)
      regions(id)%ptr = regions(id)%ptr + 1
      regions(id)%num_calls = regions(id)%num_calls + 1

      if (regions(id)%depth >= SP_MAX_DEPTH) then
         write (error_unit, "(a,a,a,i0,a)") "scope_profiler: region '", &
            trim(regions(id)%name), "' nested deeper than ", SP_MAX_DEPTH, &
            "; this call is not timed"
         return
      end if
      regions(id)%depth = regions(id)%depth + 1
      regions(id)%open_slots(regions(id)%depth) = regions(id)%ptr
      regions(id)%start_times(regions(id)%ptr) = sp_now_ns()
   end subroutine sp_begin

   !> Leave a region, writing the end time into the slot reserved by sp_begin.
   subroutine sp_end(id)
      integer, intent(in) :: id
      integer(int64) :: slot

      if (.not. active) return
      if (id < 1 .or. id > n_regions) return
      if (regions(id)%depth <= 0) then
         write (error_unit, "(a,a,a)") "scope_profiler: sp_end('", &
            trim(regions(id)%name), "') without a matching sp_begin"
         return
      end if

      slot = regions(id)%open_slots(regions(id)%depth)
      regions(id)%depth = regions(id)%depth - 1
      regions(id)%end_times(slot) = sp_now_ns()
   end subroutine sp_end

   !> sp_begin() for callers that would rather pass the name every time.
   subroutine sp_begin_name(name)
      character(len=*), intent(in) :: name
      call sp_begin(sp_region(name))
   end subroutine sp_begin_name

   !> sp_end() for callers that would rather pass the name every time.
   subroutine sp_end_name(name)
      character(len=*), intent(in) :: name
      call sp_end(sp_region(name))
   end subroutine sp_end_name

   !> Number of times a region was entered so far (0 for an unknown handle).
   !!
   !! Readable after sp_finalize() too, matching the Python API, where a
   !! region's call count outlives the run it was recorded in.
   function sp_num_calls(id) result(calls)
      integer, intent(in) :: id
      integer(int64) :: calls

      calls = 0_int64
      if (.not. allocated(regions)) return
      if (id < 1 .or. id > n_regions) return
      calls = regions(id)%num_calls
   end function sp_num_calls

   !> Double a region's timestamp buffers, keeping every reserved slot valid.
   subroutine grow(id)
      integer, intent(in) :: id
      integer(int64) :: new_capacity
      integer(int64), allocatable :: bigger(:)

      new_capacity = max(1_int64, 2_int64*regions(id)%capacity)

      allocate (bigger(new_capacity))
      bigger(1:regions(id)%capacity) = regions(id)%start_times
      call move_alloc(bigger, regions(id)%start_times)

      allocate (bigger(new_capacity))
      bigger(1:regions(id)%capacity) = regions(id)%end_times
      call move_alloc(bigger, regions(id)%end_times)

      regions(id)%capacity = new_capacity
   end subroutine grow

   !> Write this rank's trace and stop profiling.
   !!
   !! Produces `<prefix>_rank<NNNNN>.spt`. Regions that were never entered are
   !! skipped, so the file holds exactly what was measured. Anything still open
   !! is reported and dropped rather than written with a missing end time.
   subroutine sp_finalize()
      integer :: unit, i, ios, name_len
      integer(int64) :: written_regions

      if (.not. active) return

      do i = 1, n_regions
         if (regions(i)%depth /= 0) then
            write (error_unit, "(a,a,a,i0,a)") "scope_profiler: region '", &
               trim(regions(i)%name), "' still open at sp_finalize (depth ", &
               regions(i)%depth, "); its last call(s) are dropped"
            ! The reserved slots have no end time; do not write them out.
            regions(i)%ptr = regions(i)%open_slots(1) - 1
         end if
      end do

      written_regions = 0
      do i = 1, n_regions
         if (regions(i)%ptr > 0) written_regions = written_regions + 1
      end do

      open (newunit=unit, file=trace_path(), access="stream", &
            form="unformatted", status="replace", action="write", iostat=ios)
      if (ios /= 0) then
         write (error_unit, "(a,a)") "scope_profiler: cannot write ", trace_path()
         active = .false.
         return
      end if

      write (unit) SP_MAGIC
      write (unit) SP_FORMAT_VERSION
      write (unit) int(rank_id, int32)
      write (unit) written_regions

      do i = 1, n_regions
         if (regions(i)%ptr <= 0) cycle
         name_len = len_trim(regions(i)%name)
         write (unit) int(name_len, int32)
         write (unit) regions(i)%name(1:name_len)
         write (unit) regions(i)%ptr
         write (unit) regions(i)%start_times(1:regions(i)%ptr)
         write (unit) regions(i)%end_times(1:regions(i)%ptr)
      end do

      close (unit)
      active = .false.
   end subroutine sp_finalize

   !> `<prefix>_rank<NNNNN>.spt`
   function trace_path() result(path)
      character(len=len_trim(output_prefix) + 16) :: path
      write (path, "(a,a,i5.5,a)") trim(output_prefix), "_rank", rank_id, ".spt"
   end function trace_path

end module scope_profiler
