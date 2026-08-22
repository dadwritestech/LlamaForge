# syntax=docker/dockerfile:1
#
# LlamaForge has no official Dockerfile/image - this builds one from source.
# It bakes in everything LlamaForge would otherwise ask you to install via its
# Setup tab (git, cmake, ninja, a C/C++ toolchain), so that tab's installer
# prompts are unnecessary in this container.
#
# GPU build (default): nvidia/cuda devel base, needs the NVIDIA Container
# Toolkit on the host. CPU-only build: override BASE_IMAGE with ubuntu:22.04
# (via DOCKER_BASE_IMAGE in .env) and set ENABLE_CUDA=false at runtime.

ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG LLAMAFORGE_REPO=https://github.com/CaspervanWetten/LlamaForge.git
ARG LLAMAFORGE_REF=master

ENV DEBIAN_FRONTEND=noninteractive \
    APP_DIR=/opt/llamaforge \
    DATA_DIR=/data \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git cmake ninja-build build-essential \
        curl ca-certificates lsof procps \
    && rm -rf /var/lib/apt/lists/*

# Pull the app itself (pure-stdlib Python backend + static dashboard, no pip
# install needed for the app; the optional system tray is skipped headless).
RUN git clone --depth 1 --branch "${LLAMAFORGE_REF}" "${LLAMAFORGE_REPO}" "${APP_DIR}"

WORKDIR ${APP_DIR}
RUN chmod +x run.sh bootstrap.sh stop.sh 2>/dev/null || true

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && mkdir -p ${DATA_DIR}

# 8080 = llama.cpp router / OpenAI + Anthropic-compatible API
# 8090 = LlamaForge dashboard
EXPOSE 8080 8090

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
