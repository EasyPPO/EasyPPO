"""Configure CUDA and dependency compatibility for native critic training."""

import sysconfig
from importlib.metadata import version
from pathlib import Path


def configure_source(path, before, after):
    source = path.read_text()
    if after in source:
        return
    if source.count(before) != 1:
        raise RuntimeError(f"Unexpected dependency source version: {path}")
    path.write_text(source.replace(before, after, 1))


def configure_dependencies(site):
    te = site / "transformer_engine/pytorch"
    for relative, indent in (("module/grouped_linear.py", "        "), ("module/linear.py", "    ")):
        before = f'{indent}if backward_override == "high_precision":\n{indent}    save_original_input = True\n'
        configure_source(
            te / relative, before,
            before + f'{indent}elif backward_override == "dequantized":\n{indent}    save_original_input = False\n',
        )
    path = te / "attention/dot_product_attention/utils.py"
    before = (
        "            or (\n"
        "                fa2_padded_head_dim > 192\n"
        "                and device_compute_capability not in ((8, 0), (9, 0), (10, 0), (12, 0))\n"
        "            )\n"
    )
    source = path.read_text()
    if before in source:
        source = source.replace(before, "", 1).replace('            "(>192 requires sm80/90/100+). "\n', "", 1)
        path.write_text(source)
    elif "fa2_padded_head_dim > 192" in source:
        raise RuntimeError(f"Unexpected attention compatibility source: {path}")
    path = site / "fla/ops/kda/chunk_intra_token_parallel.py"
    source = path.read_text()
    if "    BK: tl.constexpr = triton.next_power_of_2(K)\n" in source:
        source = source.replace("    K: tl.constexpr,\n", "    K: tl.constexpr,\n    BK: tl.constexpr,\n", 1)
        source = source.replace("    BK: tl.constexpr = triton.next_power_of_2(K)\n", "", 1)
        source = source.replace("        K=K,\n", "        K=K,\n        BK=triton.next_power_of_2(K),\n", 1)
        path.write_text(source)
    elif "        BK=triton.next_power_of_2(K),\n" not in source:
        raise RuntimeError(f"Unexpected FLA compatibility source: {path}")


def main():
    site = Path(sysconfig.get_paths()["purelib"])
    for package, expected in {
        "torch": "2.13.0+cu130",
        "transformers": "5.12.1",
        "ray": "2.58.1",
        "nvidia-cuda-nvcc": "13.0.88",
        "nvidia-cuda-crt": "13.0.88",
        "nvidia-nvvm": "13.0.88",
        "nvidia-cuda-cccl": "13.0.85",
        "nvidia-cuda-runtime": "13.0.96",
        "transformer-engine": "2.17.0",
        "matplotlib": "3.11.2",
    }.items():
        if version(package) != expected:
            raise RuntimeError(f"{package}: expected {expected}, found {version(package)}")
    cuda = site / "nvidia/cu13"
    for required in (cuda / "bin/nvcc", cuda / "lib/libcudart.so.13", cuda / "include/cuda_runtime.h"):
        if not required.is_file():
            raise FileNotFoundError(required)
    for link, target in ((cuda / "lib64", Path("lib")), (cuda / "lib/libcudart.so", Path("libcudart.so.13"))):
        if link.exists() or link.is_symlink():
            if link.resolve() != (link.parent / target).resolve():
                raise RuntimeError(f"Unexpected CUDA linker path: {link}")
        else:
            link.symlink_to(target)
    configure_dependencies(site)


if __name__ == "__main__":
    main()
