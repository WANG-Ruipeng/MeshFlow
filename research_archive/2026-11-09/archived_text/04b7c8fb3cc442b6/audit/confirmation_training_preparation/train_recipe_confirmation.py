#!/usr/bin/env python3
"""Execute one explicitly registered independent-input confirmation recipe; never self-retry."""
from meshflow_control import runtime
import argparse
import json
import sys
from meshflow_control.training.recipe_confirmation import train_confirmation, RECIPES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, choices=RECIPES)
    parser.add_argument("--hybrid-schedule", choices=("all", "late"))
    parser.add_argument("--official", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--stream-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = train_confirmation(**vars(args))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "COMPLETE" else 75


if __name__ == "__main__":
    sys.exit(main())
