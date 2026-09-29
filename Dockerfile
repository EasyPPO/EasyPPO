FROM vllm/vllm-openai:v0.21.0@sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b

# CPU-only image construction; no model weights or tokenizers are downloaded.
USER root
WORKDIR /workspace/EasyPPO
RUN if ! command -v python >/dev/null 2>&1; then \
      ln -s "$(command -v python3)" /usr/local/bin/python; \
    fi && python -m pip --version
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface VLLM_CACHE_ROOT=/cache/vllm \
    TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
    PYTHONPATH=/workspace/EasyPPO
COPY docker/requirements-easyppo.txt /tmp/requirements-easyppo.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-easyppo.txt
# Build against this exact torch ABI. Default target is Modal H100/H200 (SM90).
ARG FLASH_ATTN_CUDA_ARCHS=90
ARG MAX_JOBS=2
RUN python3 -m pip install --no-cache-dir packaging ninja setuptools wheel \
    && FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS} \
       MAX_JOBS=${MAX_JOBS} NVCC_THREADS=2 \
       python3 -m pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3
# The serving image bundles optional LMCache, whose numpy cap conflicts with the
# paper stack. PPO does not use a KV-cache connector or the inherited system
# PyGObject desktop bindings (which lack a Python 3.12 pycairo dependency).
RUN python3 -m pip uninstall -y lmcache PyGObject && python3 -m pip check
# Source is supplied at container startup by Docker Compose or Modal.
RUN python3 -c "import importlib.util; assert importlib.util.find_spec('verl') is None"
ENTRYPOINT []
CMD ["python3", "scripts/check_environment.py"]
