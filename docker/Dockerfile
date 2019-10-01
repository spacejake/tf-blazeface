# Based on https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         vim-gtk
         ca-certificates \
         libboost-all-dev \
         python-qt4 \
         libjpeg-dev \
         x11-apps vim-gtk \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda config --set always_yes yes --set changeps1 no && conda update -q conda 
RUN conda install pytorch torchvision cudatoolkit=9.0 magma-cuda90 -c pytorch
RUN conda install progress protobuf=3.6.1 tqdm h5py pandas 
RUN conda install tensorflow-gpu
RUN conda install -c auto logging
RUN conda install -c conda-forge coloredlogs

WORKDIR /workspace
RUN chmod -R a+w /workspace
Add requirements.txt /workspace/
RUN pip install -r requirements.txt
