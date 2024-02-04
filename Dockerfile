FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ARG BRANCH=master

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    python3.10-venv \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6


# Clone repo
RUN git clone https://github.com/3DOM-FBK/deep-image-matching.git /workspace/dim
WORKDIR /workspace/dim

# Checkout the specified branch
RUN git checkout ${BRANCH}

# Create virtual environment
RUN python3.10 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install deep-image-matching
RUN python3 -m pip install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install torch torchvision
RUN pip3 install -e .
RUN pip3 install pycolmap

# Running the tests:
RUN python -m pytest  
