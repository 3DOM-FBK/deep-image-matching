FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ARG BRANCH=master

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Clone repo
RUN git clone https://github.com/3DOM-FBK/deep-image-matching.git /workspace/dim
WORKDIR /workspace/dim

# Checkout the specified branch
RUN git checkout ${BRANCH}

# Install deep-image-matching with uv
RUN uv sync --dev
RUN uv pip install pycolmap

# Running the tests:
RUN uv run pytest  
