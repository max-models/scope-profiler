# Build the two Fortran examples.
#
#   make                  # build both, if the tooling is there
#   make standalone       # pure Fortran program        (needs: a compiler)
#   make mixed            # f2py extension for Python   (needs: + meson, ninja)
#   make run-standalone   # run it, import the trace, print the summary
#   make run-mixed        # run the Python driver
#   make clean
#
# The profiler module itself comes from the installed package, so there is
# nothing to vendor: fortran_source_path() locates it.

FC ?= gfortran
FFLAGS ?= -O2 -Wall
PYTHON ?= python3
BUILD_DIR ?= build

SP_MODULE := $(shell $(PYTHON) -c "import scope_profiler.native_trace as t; print(t.fortran_source_path())" 2>/dev/null)

# Only a build needs the sources, so `make clean` still works when the
# package is not importable -- otherwise a moved install would strand the
# build artefacts here.
ifeq ($(strip $(SP_MODULE)),)
ifneq ($(MAKECMDGOALS),clean)
$(error Cannot locate scope_profiler.f90. Is scope-profiler installed for $(PYTHON)? Try `pip install -e .` from the repository root, or point PYTHON at the interpreter that has it: make PYTHON=/path/to/python)
endif
endif

.PHONY: all standalone mixed run-standalone run-mixed clean

all: standalone mixed

# ---------------------------------------------------------------- standalone
standalone: $(BUILD_DIR)/standalone

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/scope_profiler.o: $(SP_MODULE) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J$(BUILD_DIR) -c $(SP_MODULE) -o $@

$(BUILD_DIR)/kernels.o: kernels.f90 $(BUILD_DIR)/scope_profiler.o
	$(FC) $(FFLAGS) -J$(BUILD_DIR) -I$(BUILD_DIR) -c kernels.f90 -o $@

$(BUILD_DIR)/standalone: standalone.f90 $(BUILD_DIR)/kernels.o $(BUILD_DIR)/scope_profiler.o
	$(FC) $(FFLAGS) -I$(BUILD_DIR) standalone.f90 $(BUILD_DIR)/kernels.o $(BUILD_DIR)/scope_profiler.o -o $@

run-standalone: standalone
	cd $(BUILD_DIR) && ./standalone
	scope-profiler import-native $(BUILD_DIR) -o $(BUILD_DIR)/profiling_data.h5

# --------------------------------------------------------------------- mixed
# f2py compiles the profiler module and the kernels into one extension, which
# Python imports directly -- so both languages run in a single process.
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
EXTENSION := kernels$(EXT_SUFFIX)

mixed: $(EXTENSION)

$(EXTENSION): kernels.f90 $(SP_MODULE)
	$(PYTHON) -m numpy.f2py -c $(SP_MODULE) kernels.f90 -m kernels --quiet

run-mixed: mixed
	$(PYTHON) run_mixed.py

clean:
	rm -rf $(BUILD_DIR) figures __pycache__ kernels.*so kernels.*.so *.spt *.h5 *.mod
