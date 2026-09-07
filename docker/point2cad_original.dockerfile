FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt -y update --fix-missing && \
    apt install -y \
        gosu \
        gcc \
        g++ \
        gcc-9 \
        g++-9 \
        git \
        cmake \
        libgmp-dev \
        libmpfr-dev \
        libgmpxx4ldbl \
        libboost-dev \
        libboost-thread-dev \
        libgl1-mesa-glx \
        libxrender1 \
        libspatialindex-dev \
        software-properties-common \
        curl \
        zip \
        unzip \
        patchelf && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --upgrade setuptools wheel

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-9

# ---------- Build PyMesh with numpy<2.0 constraint ----------
ENV ROOT_PYMESH=/opt/pymesh
RUN mkdir -p ${ROOT_PYMESH} && \
    cd ${ROOT_PYMESH} && \
    git init && \
    git pull https://github.com/PyMesh/PyMesh.git && \
    git checkout 384ba882 && \
    git submodule update --init && \
    sed -i '43s|cwd="/root/PyMesh/docker/patches"|cwd="'${ROOT_PYMESH}'/docker/patches"|' \
        ${ROOT_PYMESH}/docker/patches/patch_wheel.py && \
    echo "numpy<2.0" > /tmp/numpy-constraint.txt && \
    pip install -c /tmp/numpy-constraint.txt -r ${ROOT_PYMESH}/python/requirements.txt && \
    python setup.py bdist_wheel && \
    rm -rf build_3.10 third_party/build && \
    python ${ROOT_PYMESH}/docker/patches/patch_wheel.py dist/pymesh2*.whl && \
    pip install dist/pymesh2*.whl && \
    # Patch pymesh to remove numpy.testing.Tester (removed in numpy 2.0)
    # Locate __init__.py via site-packages (without importing pymesh)
    PYMESH_INIT=$(python -c "import site; print(site.getsitepackages()[0])")/pymesh/__init__.py && \
    sed -i 's/^from numpy.testing import Tester$/# Removed: Tester dropped in numpy 2.0/' "$PYMESH_INIT" && \
    sed -i 's/^test = Tester().test$/# Removed: Tester dropped in numpy 2.0/' "$PYMESH_INIT" && \
    python -c "import pymesh; print('PyMesh OK')"

# ---------- Project dependencies (numpy<2.0 for PyMesh compatibility) ----------
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir --ignore-installed \
    "numpy<2.0" \
    scipy \
    open3d \
    pyvista \
    matplotlib \
    tqdm \
    trimesh \
    rtree

COPY entrypoint_point2cad_original.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

WORKDIR /work

RUN rm -rf /usr/local/cuda/compat
