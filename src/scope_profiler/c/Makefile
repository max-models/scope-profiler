# Build the C region API and its example.
#
#   make                       # build example into build/
#   make CC=icx                # use a different compiler
#   make BUILD_DIR=/tmp/sp     # build somewhere else
#   make run                   # build, run, and import the output
#   make SP_HDF5=1 run         # ... writing HDF5 directly instead of a trace
#   make hpp-check             # compile-check the C++ RAII header (scope_profiler.hpp)
#   make clean
#
# By default there is nothing to link beyond libc (and libm, for the example's
# sqrt): the implementation entry point is C99, with no HDF5 and no MPI.
#
# SP_HDF5=1 compiles the optional HDF5 backend in, so sp_finalize() writes
# <prefix>_rank<NNNNN>.h5 -- an ordinary profile, needing no import -- instead
# of a .spt trace. That is the one thing that gives the library a dependency:
# libhdf5 and its headers. pkg-config locates them where it knows about them
# (the serial build, not the MPI one, which would drag in mpi.h), and
# HDF5_CFLAGS/HDF5_LIBS override it where it does not:
#
#   make SP_HDF5=1 HDF5_CFLAGS=-I/opt/hdf5/include \
#        HDF5_LIBS="-L/opt/hdf5/lib -lhdf5" run

CC ?= cc
CXX ?= c++
CFLAGS ?= -std=c99 -O2 -Wall -Wextra
CXXFLAGS ?= -std=c++11 -O2 -Wall -Wextra
BUILD_DIR ?= build

SP_HDF5 ?= 0
ifeq ($(SP_HDF5),1)
HDF5_CFLAGS ?= $(shell pkg-config --cflags hdf5-serial 2>/dev/null || \
                       pkg-config --cflags hdf5 2>/dev/null)
HDF5_LIBS ?= $(shell pkg-config --libs hdf5-serial 2>/dev/null || \
                     pkg-config --libs hdf5 2>/dev/null || echo -lhdf5)
CFLAGS += -DSP_USE_HDF5 $(HDF5_CFLAGS)
CXXFLAGS += -DSP_USE_HDF5 $(HDF5_CFLAGS)
LDLIBS += $(HDF5_LIBS)
endif

LIB_OBJ := $(BUILD_DIR)/scope_profiler.o
EXAMPLE := $(BUILD_DIR)/example
HPP_CHECK_OBJ := $(BUILD_DIR)/hpp_check.o

.PHONY: all run hpp-check clean

all: $(EXAMPLE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_OBJ): scope_profiler.c scope_profiler.h scope_profiler_impl.h \
            scope_profiler_hdf5.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I. -c scope_profiler.c -o $@

$(EXAMPLE): example.c $(LIB_OBJ)
	$(CC) $(CFLAGS) -I. example.c $(LIB_OBJ) -lm $(LDLIBS) -o $@

# import-native reads both formats, so the same rule serves an SP_HDF5=1
# build -- there it merges (or, for one rank, renames) profiles that are
# already complete rather than converting traces.
run: $(EXAMPLE)
	cd $(BUILD_DIR) && ./example
	scope-profiler import-native $(BUILD_DIR) -o $(BUILD_DIR)/profiling_data.h5

# scope_profiler.hpp is a header-only wrapper (and can include the complete
# implementation in C++17 mode). Compile-check the C++11 wrapper here.
hpp-check: scope_profiler.hpp scope_profiler.h scope_profiler_impl.h \
           scope_profiler_hdf5.h | $(BUILD_DIR)
	echo '#include "scope_profiler.hpp"' > $(BUILD_DIR)/hpp_check.cpp
	$(CXX) $(CXXFLAGS) -I. -c $(BUILD_DIR)/hpp_check.cpp -o $(HPP_CHECK_OBJ)

clean:
	rm -rf $(BUILD_DIR)
