"""Write the Frontier-CS problems as a Miles prompt dataset for agentic_critic.miles_fcs.generate.

One json line per problem: {"prompt": <statement>, "metadata": {"problem_id": <judge problem id>}}.
Launch Miles with `--input-key prompt --metadata-key metadata` and WITHOUT --apply-chat-template: the episode
loop builds the chat turns itself from the raw statement.

Usage: python scripts/make_miles_fcs_data.py --problems-dir .../algorithmic/problems --out problems.jsonl
"""

import argparse
import json
from pathlib import Path

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--problems-dir", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

n = 0
with open(args.out, "w") as f:
    for d in sorted(Path(args.problems_dir).iterdir()):
        if not ((d / "config.yaml").exists() and (d / "statement.txt").exists()):
            continue
        config = yaml.safe_load((d / "config.yaml").read_text()) or {}
        if config.get("type", "default") != "default" or not any((d / "testdata").glob("*.in")):
            continue
        f.write(json.dumps({"prompt": (d / "statement.txt").read_text(), "metadata": {"problem_id": d.name}}) + "\n")
        n += 1
print(f"{n} problems -> {args.out}")
