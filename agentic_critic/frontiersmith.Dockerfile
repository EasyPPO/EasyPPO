FROM ubuntu:24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -o Acquire::Retries=5 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=5 \
        g++ python3 python3-pip python3-requests curl ca-certificates ripgrep && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
