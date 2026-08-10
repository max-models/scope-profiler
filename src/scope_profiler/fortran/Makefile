# Build the Fortran region API and its example.
#
#   make                       # build example into build/
#   make FC=ifx                # use a different compiler
#   make BUILD_DIR=/tmp/sp     # build somewhere else
#   make run                   # build, run, and import the trace
#   make clean
#
# There is nothing to link beyond libc: the module is plain Fortran 2008 and
# needs no preprocessor flags, no HDF5 and no MPI.

FC ?= gfortran
FFLAGS ?= -O2 -Wall -Wextra
BUILD_DIR ?= build

MODULE_SRC := scope_profiler.f90
EXAMPLE_SRC := example.f90

MODULE_OBJ := $(BUILD_DIR)/scope_profiler.o
EXAMPLE := $(BUILD_DIR)/example

.PHONY: all run clean

all: $(EXAMPLE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# -J puts the .mod file next to the object rather than in the source tree.
$(MODULE_OBJ): $(MODULE_SRC) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR) -c $(MODULE_SRC) -o $@

$(EXAMPLE): $(EXAMPLE_SRC) $(MODULE_OBJ)
	$(FC) $(FFLAGS) -I$(BUILD_DIR) $(EXAMPLE_SRC) $(MODULE_OBJ) -o $@

run: $(EXAMPLE)
	cd $(BUILD_DIR) && ./example
	scope-profiler import-fortran $(BUILD_DIR) -o $(BUILD_DIR)/profiling_data.h5

clean:
	rm -rf $(BUILD_DIR)
