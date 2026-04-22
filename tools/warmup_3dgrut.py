#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.gaussian_renderer.threedgrut_backend import warmup_threedgrut


def main():
    parser = argparse.ArgumentParser(description="Warm up cached 3DGRT/3DGUT runtime assets and JIT extensions.")
    parser.add_argument(
        "--backend",
        action="append",
        choices=["3dgrt", "3dgut"],
        help="Backend(s) to warm. Defaults to both.",
    )
    args = parser.parse_args()

    backends = tuple(args.backend or ("3dgrt", "3dgut"))
    result = warmup_threedgrut(backends=backends)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
