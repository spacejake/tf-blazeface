#!/bin/bash

declare src=$1
declare data_path=$2

echo "Data Path: $data_path"

xhost +local:root; \
    nvidia-docker run -it --rm --shm-size=16G \
    -e DISPLAY=$DISPLAY \
    -e "QT_X11_NO_MITSHM=1" \
    -e CUDACXX=/opt/cuda/bin/nvcc \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $src:/workspace/src:rw \
    -v $data_path:/workspace/dataset:rw \
    -v /dev/bus/usb:/dev/bus/usb \
    --device /dev/dri \
    deep-learning 


