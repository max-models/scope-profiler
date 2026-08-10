# Build the C region API and its example.
#
#   make                       # build example into build/
#   make CC=icx                # use a different compiler
#   make BUILD_DIR=/tmp/sp     # build somewhere else
#   make run                   # build, run, and import the trace
#   make clean
#
# There is nothing to link beyond libc (and libm, for the example's sqrt):
# the implementation is one C99 file, no HDF5 and no MPI.

CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra
BUILD_DIR ?= build

LIB_OBJ := $(BUILD_DIR)/scope_profiler.o
EXAMPLE := $(BUILD_DIR)/example

.PHONY: all run clean

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

clean:
	rm -rf $(BUILD_DIR)
