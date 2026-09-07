FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        gosu \
        wget \
        git \
        cmake \
        build-essential \
        libcgal-dev \
        libeigen3-dev \
        libgmp-dev \
        libmpfr-dev \
        libboost-dev \
        libgl1-mesa-glx \
        libxrender1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- Conda environment ----------
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda tos accept && \
    conda install -c conda-forge python=3.10.20 pythonocc-core=7.9.3 -y && \
    conda clean -afy

# ---------- libigl with CGAL copyleft ----------
RUN pip install --no-cache-dir libigl==2.6.2
RUN python -c "import igl; print('libigl OK')" && \
    python -c "from igl.copyleft.cgal import remesh_self_intersections; print('CGAL copyleft OK')"

# ---------- Project dependencies ----------
RUN pip install --no-cache-dir \
    torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir \
    numpy==2.2.6 \
    scipy==1.15.3 \
    open3d==0.19.0 \
    pyvista==0.47.3 \
    matplotlib==3.10.8 \
    tqdm==4.67.3 \
    trimesh==4.11.5 \
    rtree==1.4.1 \
    h5py==3.16.0 \
    seaborn==0.13.2

# Verify imports
RUN python -c "from OCC.Core.gp import gp_Pnt; print('OCC OK')"

COPY entrypoint_point2step.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

WORKDIR /work

RUN rm -rf /usr/local/cuda/compat
