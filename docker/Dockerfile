FROM nvcr.io/nvidia/tensorflow:23.07-tf2-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
	ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install tensorflow==2.12.0 --upgrade
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
RUN pip install \
    onnx \
    onnx_tf \
    tensorflow-probability==0.12.0 \
    tf-utils keras==2.12.0 \
    matplotlib \
    opencv-python \
    onnxruntime \
    pycuda \
    colored
RUN pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com