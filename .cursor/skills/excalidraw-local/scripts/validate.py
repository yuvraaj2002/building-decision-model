#!/usr/bin/env python3
"""Validate Excalidraw JSON for common load failures."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED = [
    "id", "type", "x", "y", "width", "height", "angle", "strokeColor",
    "backgroundColor", "fillStyle", "strokeWidth", "strokeStyle", "roughness",
    "opacity", "groupIds", "frameId", "index", "roundness", "seed", "version",
    "versionNonce", "isDeleted", "boundElements", "updated", "link", "locked",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument(
        "--fix-pending-images",
        action="store_true",
        help="Set image status pending → saved and rewrite file",
    )
    args = ap.parse_args()
    path = Path(args.path)
    data = json.loads(path.read_text())
    els = data.get("elements") or []
    files = data.get("files") or {}
    errors: list[str] = []
    warnings: list[str] = []

    ids = [e.get("id") for e in els]
    for i, c in Counter(ids).items():
        if c > 1:
            errors.append(f"duplicate id: {i}")

    indexes = [e.get("index") for e in els]
    for i, c in Counter(indexes).items():
        if c > 1:
            errors.append(f"duplicate index: {i}")

    for e in els:
        if e.get("isDeleted"):
            continue
        for k in REQUIRED:
            if k not in e:
                errors.append(f"{e.get('id')}: missing {k}")
        if e.get("type") == "text":
            for k in ("text", "fontSize", "fontFamily", "originalText", "lineHeight"):
                if k not in e:
                    errors.append(f"{e.get('id')}: text missing {k}")
        if e.get("type") == "arrow":
            pts = e.get("points") or []
            if len(pts) < 2:
                errors.append(f"{e.get('id')}: arrow needs ≥2 points")
        if e.get("type") == "image":
            status = e.get("status")
            fid = e.get("fileId")
            if status == "pending":
                msg = f"image {e.get('id')} status=pending (often breaks load)"
                if args.fix_pending_images:
                    e["status"] = "saved"
                    warnings.append(msg + " → fixed to saved")
                else:
                    errors.append(msg + "; re-run with --fix-pending-images")
            if fid and fid not in files:
                errors.append(f"image {e.get('id')} missing files[{fid}]")

    if args.fix_pending_images:
        path.write_text(json.dumps(data, indent=2) + "\n")

    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        print(f"FAIL ({len(errors)} errors)", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
