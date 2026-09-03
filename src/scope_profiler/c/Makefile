# Build the C region API and its example.
#
#   make                       # build example into build/
#   make CC=icx                # use a different compiler
#   make BUILD_DIR=/tmp/sp     # build somewhere else
#   make run                   # build, run, and import the trace
#   make hpp-check             # compile-check the C++ RAII header (scope_profiler.hpp)
#   make clean
#
# There is nothing to link beyond libc (and libm, for the example's sqrt):
# the implementation is one C99 file, no HDF5 and no MPI.

CC ?= cc
CXX ?= c++
CFLAGS ?= -std=c99 -O2 -Wall -Wextra
CXXFLAGS ?= -std=c++11 -O2 -Wall -Wextra
BUILD_DIR ?= build

LIB_OBJ := $(BUILD_DIR)/scope_profiler.o
EXAMPLE := $(BUILD_DIR)/example
HPP_CHECK_OBJ := $(BUILD_DIR)/hpp_check.o

.PHONY: all run hpp-check clean

all: $(EXAMPLE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_OBJ): scope_profiler.c scope_profiler.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I. -c scope_profiler.c -o $@

$(EXAMPLE): example.c $(LIB_OBJ)
	$(CC) $(CFLAGS) -I. example.c $(LIB_OBJ) -lm -o $@

run: $(EXAMPLE)
	cd $(BUILD_DIR) && ./example
	scope-profiler import-native $(BUILD_DIR) -o $(BUILD_DIR)/profiling_data.h5

# scope_profiler.hpp is header-only; there is nothing to link, just a
# translation unit that includes it, so a stale copy that fails to compile is
# caught here rather than at a user's first build.
hpp-check: scope_profiler.hpp scope_profiler.h | $(BUILD_DIR)
	echo '#include "scope_profiler.hpp"' > $(BUILD_DIR)/hpp_check.cpp
	$(CXX) $(CXXFLAGS) -I. -c $(BUILD_DIR)/hpp_check.cpp -o $(HPP_CHECK_OBJ)

clean:
	rm -rf $(BUILD_DIR)
