# EasyPPO container

Build the release image from the repository root:

```bash
docker build --platform linux/amd64 -t easyppo:modal .
docker run --rm --network none easyppo:modal
```

The release uses the root [Dockerfile](../Dockerfile) and
[`requirements-easyppo.txt`](requirements-easyppo.txt). The default command
checks the environment on CPU without loading a model. The image targets
H100/H200 GPUs; its first build compiles FlashAttention for SM90.

The [project README](../README.md) covers the AIME recipe, Modal setup,
training entrypoint, and checkpoint downloads.
