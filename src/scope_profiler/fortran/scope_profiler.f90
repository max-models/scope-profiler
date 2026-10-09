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
!! uses - `CLOCK_MONOTONIC` on Linux, `CLOCK_UPTIME_RAW` on macOS - so
!! regions recorded here share an epoch with regions recorded by the Python
!! API in the same process tree, and land on one timeline. The right clock is
!! found by probing at run time, so the file is plain Fortran needing no
!! preprocessor and no platform flags.
module scope_profiler
   use, intrinsic :: iso_c_binding, only: c_int, c_long
   use, intrinsic :: iso_fortran_env, only: int32, int64, error_unit
   implicit none
   private

   public :: sp_init, sp_region, sp_begin, sp_end
   public :: sp_begin_name, sp_end_name, sp_finalize
   public :: sp_num_calls, sp_now_ns, sp_is_active
   public :: sp_flush, sp_reset, sp_get_region_stats

   integer, parameter, public :: SP_OK = 0, SP_ERR_INACTIVE = 1, SP_ERR_NO_CLOCK = 2
   integer, parameter, public :: SP_ERR_NO_MEMORY = 3, SP_ERR_IO = 4
   integer, parameter, public :: SP_ERR_UNMATCHED_END = 5, SP_ERR_OPEN_SCOPES = 6
   integer, parameter, public :: SP_ERR_INVALID_ARGUMENT = 7, SP_ERR_DEPTH = 8

   !> Statistics of completed calls, computed only when requested. All times
   !! are nanoseconds; every field is zero when there are no completed calls.
   type, public :: sp_region_stats
      integer(int64) :: calls = 0, total_ns = 0, min_ns = 0, max_ns = 0
   end type sp_region_stats

   !> Longest region name the trace format stores.
   integer, parameter, public :: SP_MAX_NAME = 128

   !> Trace format written by sp_finalize; keep in step with native_trace.py.
   character(len=8), parameter :: SP_MAGIC = "SCOPEPRF"
   integer(int32), parameter :: SP_FORMAT_VERSION = 2

   !> Initial slots per region; the buffers double from here as needed.
   integer, parameter :: SP_INITIAL_CAPACITY = 1024

   !> Deepest recursion of a single region that can be open at once.
   integer, parameter :: SP_MAX_DEPTH = 64

   type :: region_t
      character(len=SP_MAX_NAME) :: name = ""
      character(len=:), allocatable :: source_file
      integer(int32) :: source_line = -1
      integer(int64), allocatable :: start_times(:)
      integer(int64), allocatable :: end_times(:)
      integer(int64) :: ptr = 0            !< slots used
      integer(int64) :: capacity = 0
      integer(int64) :: num_calls = 0
      !> Slots reserved by regions still open, innermost last. A recursive
      !! re-entry reserves its own slot instead of overwriting the outer one.
      integer(int64) :: open_slots(SP_MAX_DEPTH) = 0
      integer :: depth = 0
      ! Rejected entries must consume their own ends, not an outer call's.
      integer :: skipped_depth = 0
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
   !! sp_init() reports and refuses to profile with - silently handing back 0
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

   !> Optional status outputs are set on every call. Errors are printed only
   !! when neither output is supplied. Inactive instrumentation stays quiet.
   subroutine report(code, message, stat, errmsg)
      integer, intent(in) :: code
      character(len=*), intent(in) :: message
      integer, intent(out), optional :: stat
      character(len=*), intent(out), optional :: errmsg

      if (present(stat)) stat = code
      if (present(errmsg)) errmsg = message
      if (code /= SP_OK .and. code /= SP_ERR_INACTIVE) then
         if (.not. present(stat) .and. .not. present(errmsg)) &
            write (error_unit, '(a)') 'scope_profiler: '//message
      end if
   end subroutine report

   !> Start a fresh session; existing handles and recordings are replaced.
   !! Prefixes longer than 512 characters and negative ranks are rejected.
   subroutine sp_init(prefix, rank, stat, errmsg)
      character(len=*), intent(in) :: prefix
      integer, intent(in), optional :: rank
      integer, intent(out), optional :: stat
      character(len=*), intent(out), optional :: errmsg
      integer :: ios
      character(len=512) :: message

      call report(SP_OK, '', stat, errmsg)
      if (len_trim(prefix) > len(output_prefix) .or. len_trim(prefix) == 0) then
         call report(SP_ERR_INVALID_ARGUMENT, 'invalid output prefix length', stat, errmsg)
         return
      end if
      if (present(rank)) then
         if (rank < 0) then
            call report(SP_ERR_INVALID_ARGUMENT, 'rank must be nonnegative', stat, errmsg)
            return
         end if
      end if
      active = .false.
      call resolve_clock()
      if (clock_id < 0_c_int) then
         call report(SP_ERR_NO_CLOCK, 'no monotonic clock available', stat, errmsg)
         return
      end if
      if (allocated(regions)) deallocate (regions)
      n_regions = 0
      allocate (regions(16), stat=ios, errmsg=message)
      if (ios /= 0) then
         call report(SP_ERR_NO_MEMORY, trim(message), stat, errmsg)
         return
      end if
      output_prefix = prefix
      rank_id = 0
      if (present(rank)) rank_id = rank
      active = .true.
   end subroutine sp_init

   !> Resolve a name once, outside hot loops. Names retain the historical
   !! 128-character truncation, including during lookup. Source metadata is
   !! first-writer-wins, and may be backfilled after registration without it.
   function sp_region(name, file, line, stat, errmsg) result(id)
      character(len=*), intent(in) :: name
      character(len=*), intent(in), optional :: file
      integer, intent(in), optional :: line
      integer, intent(out), optional :: stat
      character(len=*), intent(out), optional :: errmsg
      integer :: id, i, candidate, ios
      character(len=SP_MAX_NAME) :: key
      character(len=512) :: message
      type(region_t), allocatable :: bigger(:)

      id = 0
      call report(SP_OK, '', stat, errmsg)
      if (.not. active) then
         call report(SP_ERR_INACTIVE, 'profiler is inactive', stat, errmsg)
         return
      end if
      key = name
      candidate = n_regions + 1
      do i = 1, n_regions
         if (regions(i)%name == key) then
            candidate = i
            exit
         end if
      end do
      if (candidate > size(regions)) then
         allocate (bigger(2*size(regions)), stat=ios, errmsg=message)
         if (ios /= 0) then
            call report(SP_ERR_NO_MEMORY, trim(message), stat, errmsg)
            return
         end if
         ! Move owned buffers: intrinsic derived-type assignment would copy
         ! them with implicit allocations whose failures cannot be caught.
         do i = 1, n_regions
            bigger(i)%name = regions(i)%name
            bigger(i)%source_line = regions(i)%source_line
            bigger(i)%ptr = regions(i)%ptr
            bigger(i)%capacity = regions(i)%capacity
            bigger(i)%num_calls = regions(i)%num_calls
            bigger(i)%open_slots = regions(i)%open_slots
            bigger(i)%depth = regions(i)%depth
            bigger(i)%skipped_depth = regions(i)%skipped_depth
            call move_alloc(regions(i)%source_file, bigger(i)%source_file)
            call move_alloc(regions(i)%start_times, bigger(i)%start_times)
            call move_alloc(regions(i)%end_times, bigger(i)%end_times)
         end do
         call move_alloc(bigger, regions)
      end if
      if (candidate > n_regions .and. regions(candidate)%capacity == 0) then
         call grow(candidate, ios, message)
         if (ios /= SP_OK) then
            call report(ios, trim(message), stat, errmsg)
            return
         end if
      end if
      if (present(file)) then
         if (len_trim(file) > 0 .and. .not. allocated(regions(candidate)%source_file)) then
            allocate (character(len=len_trim(file)) :: regions(candidate)%source_file, stat=ios, errmsg=message)
            if (ios /= 0) then
               call report(SP_ERR_NO_MEMORY, trim(message), stat, errmsg)
               return
            end if
            regions(candidate)%source_file(:) = trim(file)
            if (present(line)) then
               if (line >= 0) regions(candidate)%source_line = int(line, int32)
            end if
         end if
      end if
      regions(candidate)%name = key
      n_regions = max(n_regions, candidate)
      id = candidate
   end function sp_region

   !> Enter a region. Failed entries consume their matching sp_end without
   !! closing an outer invocation of the same region.
   subroutine sp_begin(id, stat, errmsg)
      integer, intent(in) :: id
      integer, intent(out), optional :: stat
      character(len=*), intent(out), optional :: errmsg
      integer :: ios
      integer(int64) :: now, slot
      character(len=512) :: message

      if (present(stat)) stat = SP_OK
      if (present(errmsg)) errmsg = ''
      if (.not. active) then
         call report(SP_ERR_INACTIVE, 'profiler is inactive', stat, errmsg)
         return
      end if
      if (id < 1 .or. id > n_regions) then
         call report(SP_ERR_INVALID_ARGUMENT, 'invalid region handle', stat, errmsg)
         return
      end if
      if (regions(id)%depth >= SP_MAX_DEPTH .or. regions(id)%skipped_depth > 0) then
         regions(id)%skipped_depth = regions(id)%skipped_depth + 1
         call report(SP_ERR_DEPTH, 'region nesting limit exceeded or enclosing entry failed', stat, errmsg)
         return
      end if
      if (regions(id)%ptr >= regions(id)%capacity) then
         call grow(id, ios, message)
         if (ios /= SP_OK) then
            regions(id)%skipped_depth = regions(id)%skipped_depth + 1
            call report(ios, trim(message), stat, errmsg)
            return
         end if
      end if
      now = sp_now_ns()
      if (now < 0) then
         regions(id)%skipped_depth = regions(id)%skipped_depth + 1
         call report(SP_ERR_NO_CLOCK, 'cannot read monotonic clock', stat, errmsg)
         return
      end if
      slot = regions(id)%ptr + 1
      regions(id)%ptr = slot
      regions(id)%num_calls = regions(id)%num_calls + 1
      regions(id)%depth = regions(id)%depth + 1
      regions(id)%open_slots(regions(id)%depth) = slot
      regions(id)%start_times(slot) = now
      regions(id)%end_times(slot) = -1_int64
   end subroutine sp_begin

   !> Leave a region. No statistics are accumulated on this hot path.
   subroutine sp_end(id, stat, errmsg)
      integer, intent(in) :: id
      integer, intent(out), optional :: stat
      character(len=*), intent(out), optional :: errmsg
      integer(int64) :: slot, now

      if (present(stat)) stat = SP_OK
      if (present(errmsg)) errmsg = ''
      if (.not. active) then
         call report(SP_ERR_INACTIVE, 'profiler is inactive', stat, errmsg)
         return
      end if
      if (id < 1 .or. id > n_regions) then
         call report(SP_ERR_INVALID_ARGUMENT, 'invalid region handle', stat, errmsg)
         return
      end if
      if (regions(id)%skipped_depth > 0) then
         regions(id)%skipped_depth = regions(id)%skipped_depth - 1
         return
      end if
      if (regions(id)%depth <= 0) then
         call report(SP_ERR_UNMATCHED_END, 'sp_end without a matching sp_begin', stat, errmsg)
         return
      end if
      slot = regions(id)%open_slots(regions(id)%depth)
      regions(id)%depth = regions(id)%depth - 1
      now = sp_now_ns()
      if (now < 0) then
         call report(SP_ERR_NO_CLOCK, 'cannot read monotonic clock; call dropped', stat, errmsg)
         return
      end if
      regions(id)%end_times(slot) = now
   end subroutine sp_end

   subroutine sp_begin_name(name, stat, errmsg)
      character(len=*), intent(in) :: name
      integer, intent(out), optional :: stat
      character(len=*), intent(out), optional :: errmsg
      integer :: id, code
      character(len=512) :: message

      id = sp_region(name, stat=code, errmsg=message)
      if (code /= SP_OK) then
         call report(code, trim(message), stat, errmsg)
         return
      end if
      call sp_begin(id, stat, errmsg)
   end subroutine sp_begin_name

   subroutine sp_end_name(name, stat, errmsg)
      character(len=*), intent(in) :: name
      integer, intent(out), optional :: stat
      character(len=*), intent(out), optional :: errmsg
      integer :: i
      character(len=SP_MAX_NAME) :: key

      if (.not. active) then
         call report(SP_ERR_INACTIVE, 'profiler is inactive', stat, errmsg)
         return
      end if
      key = name
      do i = 1, n_regions
         if (regions(i)%name /= key) cycle
         call sp_end(i, stat, errmsg)
         return
      end do
      call report(SP_ERR_UNMATCHED_END, 'sp_end_name without a matching sp_begin', stat, errmsg)
   end subroutine sp_end_name

   !> Successful entries, including still-open calls. Readable after finalize.
   function sp_num_calls(id) result(calls)
      integer, intent(in) :: id
      integer(int64) :: calls
      calls = 0_int64
      if (.not. allocated(regions)) return
      if (id < 1 .or. id > n_regions) return
      calls = regions(id)%num_calls
   end function sp_num_calls

   !> Compute inclusive duration statistics in O(recorded calls), ignoring
   !! open/dropped calls. Also readable after finalization.
   subroutine sp_get_region_stats(id, stats, stat, errmsg)
      integer, intent(in) :: id
      type(sp_region_stats), intent(out) :: stats
      integer, intent(out), optional :: stat
      character(len=*), intent(out), optional :: errmsg
      integer(int64) :: j, duration

      stats = sp_region_stats()
      call report(SP_OK, '', stat, errmsg)
      if (.not. allocated(regions)) then
         call report(SP_ERR_INACTIVE, 'profiler has not been initialized', stat, errmsg)
         return
      end if
      if (id < 1 .or. id > n_regions) then
         call report(SP_ERR_INVALID_ARGUMENT, 'invalid region handle', stat, errmsg)
         return
      end if
      do j = 1, regions(id)%ptr
         if (regions(id)%end_times(j) < 0) cycle
         duration = regions(id)%end_times(j) - regions(id)%start_times(j)
         if (stats%calls == 0) stats%min_ns = duration
         stats%calls = stats%calls + 1
         stats%total_ns = stats%total_ns + duration
         stats%min_ns = min(stats%min_ns, duration)
         stats%max_ns = max(stats%max_ns, duration)
      end do
   end subroutine sp_get_region_stats

   !> Clear measurements, retaining region handles, metadata and capacity.
   !! Refuse without changing anything if any invocation is still open.
   subroutine sp_reset(stat, errmsg)
      integer, intent(out), optional :: stat
      character(len=*), intent(out), optional :: errmsg
      integer :: i
      call report(SP_OK, '', stat, errmsg)
      if (.not. active) then
         call report(SP_ERR_INACTIVE, 'profiler is inactive', stat, errmsg)
         return
      end if
      do i = 1, n_regions
         if (regions(i)%depth == 0 .and. regions(i)%skipped_depth == 0) cycle
         call report(SP_ERR_OPEN_SCOPES, 'cannot reset with open regions', stat, errmsg)
         return
      end do
      do i = 1, n_regions
         regions(i)%ptr = 0
         regions(i)%num_calls = 0
      end do
   end subroutine sp_reset

   !> Allocate both buffers before replacing either, preserving old data on
   !! failure. Only used at registration and when a buffer fills.
   subroutine grow(id, stat, errmsg)
      integer, intent(in) :: id
      integer, intent(out) :: stat
      character(len=*), intent(out) :: errmsg
      integer :: ios
      integer(int64) :: capacity, used
      integer(int64), allocatable :: starts(:), ends(:)

      stat = SP_OK
      errmsg = ''
      capacity = max(int(SP_INITIAL_CAPACITY, int64), 2_int64*regions(id)%capacity)
      allocate (starts(capacity), ends(capacity), stat=ios, errmsg=errmsg)
      if (ios /= 0) then
         stat = SP_ERR_NO_MEMORY
         return
      end if
      used = regions(id)%ptr
      if (used > 0) then
         starts(1:used) = regions(id)%start_times(1:used)
         ends(1:used) = regions(id)%end_times(1:used)
      end if
      call move_alloc(starts, regions(id)%start_times)
      call move_alloc(ends, regions(id)%end_times)
      regions(id)%capacity = capacity
   end subroutine grow

   !> Snapshot all completed calls, without stopping or discarding data.
   !! Replaces the previous snapshot at the same path; not an append or an
   !! atomic checkpoint. Open calls are excluded, including recursive ones.
   subroutine sp_flush(stat, errmsg)
      integer, intent(out), optional :: stat
      character(len=*), intent(out), optional :: errmsg
      integer :: unit, i, ios, close_ios, name_len, source_len
      integer(int64) :: written_regions, completed, j, k, max_calls, expected_size, actual_size
      integer(int64), allocatable :: starts(:), ends(:)
      character(len=512) :: message, close_message

      call report(SP_OK, '', stat, errmsg)
      if (.not. active) then
         call report(SP_ERR_INACTIVE, 'profiler is inactive', stat, errmsg)
         return
      end if
      written_regions = 0
      max_calls = 0
      do i = 1, n_regions
         completed = count(regions(i)%end_times(1:regions(i)%ptr) >= 0, kind=int64)
         if (completed > 0) written_regions = written_regions + 1
         max_calls = max(max_calls, completed)
      end do
      ! Scratch allocation happens before opening/truncating the output.
      allocate (starts(max_calls), ends(max_calls), stat=ios, errmsg=message)
      if (ios /= 0) then
         call report(SP_ERR_NO_MEMORY, trim(message), stat, errmsg)
         return
      end if
      open (newunit=unit, file=trace_path(), access='stream', form='unformatted', &
            status='replace', action='write', iostat=ios, iomsg=message)
      if (ios /= 0) then
         call report(SP_ERR_IO, 'cannot write '//trace_path()//': '//trim(message), stat, errmsg)
         return
      end if
      expected_size = 24_int64
      write (unit, iostat=ios, iomsg=message) SP_MAGIC, SP_FORMAT_VERSION, int(rank_id, int32), written_regions
      if (ios == 0) then
         do i = 1, n_regions
            k = 0
            do j = 1, regions(i)%ptr
               if (regions(i)%end_times(j) < 0) cycle
               k = k + 1
               starts(k) = regions(i)%start_times(j)
               ends(k) = regions(i)%end_times(j)
            end do
            if (k == 0) cycle
            name_len = len_trim(regions(i)%name)
            source_len = 0
            if (allocated(regions(i)%source_file)) source_len = len(regions(i)%source_file)
            write (unit, iostat=ios, iomsg=message) int(name_len, int32), &
               regions(i)%name(1:name_len), int(source_len, int32)
            if (ios /= 0) exit
            if (source_len > 0) then
               write (unit, iostat=ios, iomsg=message) regions(i)%source_file
               if (ios /= 0) exit
            end if
            expected_size = expected_size + 20_int64 + name_len + source_len + 16_int64*k
            write (unit, iostat=ios, iomsg=message) regions(i)%source_line, k, starts(1:k), ends(1:k)
            if (ios /= 0) exit
         end do
      end if
      ! Force buffered writes now: some runtimes do not propagate a failed
      ! buffer drain through CLOSE's IOSTAT.
      if (ios == 0) flush (unit, iostat=ios, iomsg=message)
      close (unit, iostat=close_ios, iomsg=close_message)
      if (ios /= 0) then
         call report(SP_ERR_IO, 'cannot write '//trace_path()//': '//trim(message), stat, errmsg)
      else if (close_ios /= 0) then
         call report(SP_ERR_IO, 'cannot close '//trace_path()//': '//trim(close_message), stat, errmsg)
      else
         ! Some runtimes silently accept a partial buffered write (e.g. at
         ! RLIMIT_FSIZE). Check the closed file, not the buffered position.
         inquire (file=trace_path(), size=actual_size, iostat=ios, iomsg=message)
         if (ios /= 0) then
            call report(SP_ERR_IO, 'cannot verify '//trace_path()//': '//trim(message), stat, errmsg)
         else if (actual_size /= expected_size) then
            call report(SP_ERR_IO, 'incomplete trace written to '//trace_path(), stat, errmsg)
         end if
      end if
   end subroutine sp_flush

   !> Save completed calls and stop. Open calls are excluded but completed
   !! recursive children survive. On output failure the session stays active
   !! so the application can fix the cause and retry. Finalization is idempotent.
   subroutine sp_finalize(stat, errmsg)
      integer, intent(out), optional :: stat
      character(len=*), intent(out), optional :: errmsg
      integer :: i, code
      logical :: has_open
      character(len=512) :: message

      call report(SP_OK, '', stat, errmsg)
      if (.not. active) return
      call sp_flush(code, message)
      if (code /= SP_OK) then
         call report(code, trim(message), stat, errmsg)
         return
      end if
      has_open = .false.
      do i = 1, n_regions
         if (regions(i)%depth /= 0 .or. regions(i)%skipped_depth /= 0) has_open = .true.
      end do
      active = .false.
      if (has_open) call report(SP_ERR_OPEN_SCOPES, &
         'regions still open at sp_finalize; unfinished calls dropped', stat, errmsg)
   end subroutine sp_finalize

   function trace_path() result(path)
      character(len=:), allocatable :: path
      character(len=32) :: rank_text
      write (rank_text, '(i0.5)') rank_id
      path = trim(output_prefix)//'_rank'//trim(rank_text)//'.spt'
   end function trace_path

end module scope_profiler
