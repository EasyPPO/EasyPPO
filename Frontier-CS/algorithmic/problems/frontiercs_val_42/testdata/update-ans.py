#!/usr/bin/env python3
# update_ans.py
# Scan *.in files, compute S(n), write corresponding .ans files

import argparse
import glob
import math
import os
import re
import sys
from functools import lru_cache
from typing import Optional

# ---------- S(n) definition ----------
def s_base(n: int) -> float:
    if n == 1:
        return 1.0
    if 2 <= n <= 4:
        return 2.0
    if n == 5:
        return 2.0 + 1.0 / math.sqrt(2.0)
    if 6 <= n <= 9:
        return 3.0
    if n == 10:
        return 3.0 + 1.0 / math.sqrt(2.0)
    if n == 11:
        return 3.8771
    if 12 <= n <= 16:
        return 4.0
    if n == 17:
        return 4.6756
    if n == 18:
        return 3.5 + 0.5 * math.sqrt(7.0)
    if n == 19:
        return 3.0 + (4.0/3.0) * math.sqrt(2.0)
    if 20 <= n <= 25:
        return 5.0
    if n == 26:
        return 3.5 + 1.5 * math.sqrt(2.0)
    if n == 27:
        return 5.0 + 1.0 / math.sqrt(2.0)
    if n == 28:
        return 3.0 + 2.0 * math.sqrt(2.0)
    if n == 29:
        return 5.9344
    if 30 <= n <= 36:
        return 6.0
    if n == 37:
        return 6.5987
    if n == 38:
        return 6.0 + 1.0 / math.sqrt(2.0)
    if n == 39:
        return 6.8189
    if n == 40:
        return 4.0 + 2.0 * math.sqrt(2.0)
    if n == 41:
        return 6.9473
    if 42 <= n <= 49:
        return 7.0
    if n == 50:
        return 7.5987
    if n == 51:
        return 7.7044
    if n == 52:
        return 7.0 + 1.0 / math.sqrt(2.0)
    if n == 53:
        return 7.8231
    if n == 54:
        return 7.8488
    if n == 55:
        return 7.9871
    if 56 <= n <= 64:
        return 8.0
    if n == 65:
        return 5.0 + 5.0 / math.sqrt(2.0)
    if n == 66:
        return 3.0 + 4.0 * math.sqrt(2.0)
    if n == 67:
        return 8.0 + 1.0 / math.sqrt(2.0)
    if n == 68:
        return 7.5 + 0.5 * math.sqrt(7.0)
    if n == 69:
        return 8.8562
    if n == 70:
        return 8.9121
    if n == 71:
        return 8.9633
    if 72 <= n <= 81:
        return 9.0
    if n == 82:
        return 6.0 + 5.0 / math.sqrt(2.0)
    if n == 83:
        return 4.0 + 4.0 * math.sqrt(2.0)
    if n == 84:
        return 9.0 + 1.0 / math.sqrt(2.0)
    if n == 85:
        return 5.5 + 3.0 * math.sqrt(2.0)
    if n == 86:
        return 8.5 + 0.5 * math.sqrt(7.0)
    if n == 87:
        return 9.8520
    if n == 88:
        return 9.9018
    if n == 89:
        return 5.0 + 7.0 / math.sqrt(2.0)
    if 90 <= n <= 100:
        return 10.0
    raise ValueError("s_base only defined for 1..100")

@lru_cache(maxsize=None)
def S(n: int) -> float:
    if n <= 100:
        return s_base(n)
    return 2.0 * S((n + 3) // 4)  # ceil(n/4)

# ---------- Utilities ----------
def read_first_int(path: str) -> Optional[int]:
    """Read the first integer from a file. Allows whitespace and newlines."""
    try:
        with open(path, "r") as f:
            text = f.read()
        m = re.search(r"-?\d+", text)
        if not m:
            return None
        return int(m.group(0))
    except Exception:
        return None

def natural_key(s: str):
    """Natural sorting for human-friendly order (e.g. 2.in < 10.in)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def main():
    ap = argparse.ArgumentParser(description="Update .ans from .in by computing S(n).")
    ap.add_argument("--pattern", default="*.in", help="Glob pattern for input files (default: *.in)")
    ap.add_argument("--fmt", default=".10g", help="Float format for S(n), e.g. .10g / .6f (default: .10g)")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be written, do not write files")
    ap.add_argument("--strict", action="store_true", help="Strict mode: fail on any unreadable .in")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern), key=natural_key)
    if not files:
        print(f"No files match pattern: {args.pattern}", file=sys.stderr)
        sys.exit(1)

    wrote = 0
    failed = 0
    for in_path in files:
        n = read_first_int(in_path)
        if n is None:
            msg = f"[skip] Cannot parse integer from {in_path}"
            if args.strict:
                print(msg, file=sys.stderr)
                sys.exit(2)
            else:
                print(msg, file=sys.stderr)
                failed += 1
                continue

        try:
            val = S(n)
        except Exception as e:
            msg = f"[skip] S(n) failed for {in_path} (n={n}): {e}"
            if args.strict:
                print(msg, file=sys.stderr)
                sys.exit(3)
            else:
                print(msg, file=sys.stderr)
                failed += 1
                continue

        out_path = re.sub(r"\.in$", ".ans", in_path)
        if out_path == in_path:
            out_path = in_path + ".ans"

        txt = format(val, args.fmt) + "\n"
        if args.dry_run:
            print(f"[dry] {in_path} -> {out_path}: {txt.strip()}")
        else:
            with open(out_path, "w") as f:
                f.write(txt)
            print(f"[ok] {in_path} -> {out_path}")
            wrote += 1

    print(f"\nDone. Wrote {wrote} .ans file(s). Skipped/failed: {failed}.")

if __name__ == "__main__":
    main()