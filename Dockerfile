FROM pytorch/pytorch:latest

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update && apt install -y git
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Clone repo
RUN git clone https://github.com/3DOM-FBK/deep-image-matching.git /workspace/deep-image-matching
WORKDIR /workspace/deep-image-matching

# Install deep-image-matching
RUN pip install -e .
RUN pip install pycolmap

# Running the tests:
RUN pytest