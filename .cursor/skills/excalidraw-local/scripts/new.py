#!/usr/bin/env python3
"""Create an empty Excalidraw document (refuses to overwrite unless --force)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _excalidraw import EMPTY_DOC, save
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"EXISTS {path} (pass --force to overwrite)", file=sys.stderr)
        return 1
    save(path, json.loads(json.dumps(EMPTY_DOC)))
    print(f"created {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
