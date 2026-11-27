"""Backport the FA4 opt-out to the pinned image's Transformer Engine build.

TE 2.17.0+miles ignores NVTE_FLASH_ATTN_V4 and passes an incompatible causal
window sentinel to flash-attn-4 4.0.0b35. See NVIDIA/TransformerEngine#3528.
Keep FA4 installed for other consumers; disable it only in TE via the image env.
"""

from importlib.metadata import distribution
from pathlib import Path


def main():
    path = Path(
        distribution("transformer-engine").locate_file(
            "transformer_engine/pytorch/attention/dot_product_attention/utils.py"
        )
    )
    source = path.read_text()
    old = "    use_flash_attention_4 = use_flash_attention\n"
    new = '    use_flash_attention_4 = use_flash_attention and int(os.getenv("NVTE_FLASH_ATTN_V4", "1"))\n'
    if new in source:
        return
    if source.count(old) != 1:
        raise RuntimeError("Transformer Engine backend selection changed; review the FA4 opt-out patch")
    path.write_text(source.replace(old, new))


if __name__ == "__main__":
    main()
