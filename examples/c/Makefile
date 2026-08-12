# Build the two C examples.
#
#   make                  # build both
#   make standalone       # pure C program            (needs: a C compiler)
#   make mixed            # shared library for ctypes (needs: a C compiler)
#   make run-standalone   # run it, import the trace, print the summary
#   make run-mixed        # run the Python driver
#   make clean
#
# The profiler sources come from the installed package, so there is nothing to
# vendor: c_source_path() and c_include_dir() locate them.

CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall
PYTHON ?= python3
BUILD_DIR ?= build

SP_SOURCE := $(shell $(PYTHON) -c "import scope_profiler.native_trace as t; print(t.c_source_path())" 2>/dev/null)
SP_INCLUDE := $(shell $(PYTHON) -c "import scope_profiler.native_trace as t; print(t.c_include_dir())" 2>/dev/null)

# Only a build needs the sources, so `make clean` still works when the
# package is not importable -- otherwise a moved install would strand the
# build artefacts here.
ifeq ($(strip $(SP_SOURCE)),)
ifneq ($(MAKECMDGOALS),clean)
$(error Cannot locate scope_profiler.c. Is scope-profiler installed for \
$(PYTHON)? Try `pip install -e .` from the repository root, or point PYTHON \
at the interpreter that has it: make PYTHON=/path/to/python)
endif
endif

INCLUDES := -I. -I$(SP_INCLUDE)

.PHONY: all standalone mixed run-standalone run-mixed clean

all: standalone mixed

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/scope_profiler.o: $(SP_SOURCE) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $(SP_SOURCE) -o $@

$(BUILD_DIR)/kernels.o: kernels.c kernels.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c kernels.c -o $@

# ---------------------------------------------------------------- standalone
standalone: $(BUILD_DIR)/standalone

$(BUILD_DIR)/standalone: standalone.c $(BUILD_DIR)/kernels.o $(BUILD_DIR)/scope_profiler.o
	$(CC) $(CFLAGS) $(INCLUDES) standalone.c $(BUILD_DIR)/kernels.o \
		$(BUILD_DIR)/scope_profiler.o -lm -o $@

run-standalone: standalone
	cd $(BUILD_DIR) && ./standalone
	scope-profiler import-native $(BUILD_DIR) -o $(BUILD_DIR)/profiling_data.h5

# --------------------------------------------------------------------- mixed
# One shared library holding the kernels and the profiler, which Python loads
# with ctypes -- so both languages run in a single process. No build backend
# and no binding generator: ctypes needs nothing but the .so.
mixed: $(BUILD_DIR)/libkernels.so

$(BUILD_DIR)/libkernels.so: kernels.c $(SP_SOURCE) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -fPIC -shared kernels.c $(SP_SOURCE) -lm -o $@

run-mixed: mixed
	$(PYTHON) run_mixed.py $(BUILD_DIR)/libkernels.so

clean:
	rm -rf $(BUILD_DIR) figures __pycache__ *.spt *.h5
