# CI image for scope-profiler's GitHub Actions "Testing" workflow.
#
# Bakes in the slow, rarely-changing prerequisites that testing.yml otherwise
# rebuilds on every run: MPI, LIKWID (built from source) and pylikwid (built
# against each tested CPython, since the extension is ABI-specific). Anything
# that changes per-commit (the project itself, its Python deps) is still
# installed at test time via pip, not baked in here.
#
# Built and pushed by .github/workflows/build-ci-image.yml.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    git \
    curl \
    ca-certificates \
    software-properties-common \
    kmod \
    libhdf5-openmpi-dev \
    libopenmpi-dev \
    openmpi-bin \
    pkg-config \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    python3.12 python3.12-venv python3.12-dev \
  && rm -rf /var/lib/apt/lists/*

# LIKWID, built from source (same as the CI step it replaces).
RUN git clone --depth 1 https://github.com/RRZE-HPC/likwid.git /opt/likwid-src \
  && cd /opt/likwid-src \
  && make && make install \
  && echo "/usr/local/lib" > /etc/ld.so.conf.d/likwid.conf \
  && ldconfig \
  && rm -rf /opt/likwid-src

# pylikwid, built once per tested Python version into its own venv, since the
# compiled extension is tied to one CPython ABI.
RUN git clone --depth 1 https://github.com/RRZE-HPC/pylikwid.git /opt/pylikwid-src

RUN python3.11 -m venv /opt/venvs/py311 \
  && /opt/venvs/py311/bin/pip install --upgrade pip setuptools \
  && HDF5_MPI=ON CC=mpicc /opt/venvs/py311/bin/pip install --no-binary=h5py h5py \
  && cd /opt/pylikwid-src \
  && /opt/venvs/py311/bin/python setup.py build_ext \
       -I/usr/local/include -L/usr/local/lib -R/usr/local/lib \
  && /opt/venvs/py311/bin/python setup.py install \
  && /opt/venvs/py311/bin/python setup.py clean --all

RUN python3.12 -m venv /opt/venvs/py312 \
  && /opt/venvs/py312/bin/pip install --upgrade pip setuptools \
  && HDF5_MPI=ON CC=mpicc /opt/venvs/py312/bin/pip install --no-binary=h5py h5py \
  && cd /opt/pylikwid-src \
  && /opt/venvs/py312/bin/python setup.py build_ext \
       -I/usr/local/include -L/usr/local/lib -R/usr/local/lib \
  && /opt/venvs/py312/bin/python setup.py install \
  && /opt/venvs/py312/bin/python setup.py clean --all

RUN rm -rf /opt/pylikwid-src

RUN /opt/venvs/py311/bin/python -c "import pylikwid" \
  && /opt/venvs/py312/bin/python -c "import pylikwid" \
  && /opt/venvs/py311/bin/python -c "import h5py; assert h5py.get_config().mpi" \
  && /opt/venvs/py312/bin/python -c "import h5py; assert h5py.get_config().mpi"
