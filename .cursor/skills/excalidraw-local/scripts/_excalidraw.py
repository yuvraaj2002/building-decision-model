#!/usr/bin/env python3
"""Shared helpers for local Excalidraw mutations."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

EMPTY_DOC = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://marketplace.visualstudio.com/items?itemName=pomdtr.excalidraw-editor",
    "elements": [],
    "appState": {
        "gridSize": 20,
        "gridStep": 5,
        "gridModeEnabled": False,
        "viewBackgroundColor": "#ffffff",
    },
    "files": {},
}


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return json.loads(json.dumps(EMPTY_DOC))
    return json.loads(path.read_text())


def save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def nid() -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
    return "".join(random.choice(alphabet) for _ in range(21))


def seed() -> int:
    return random.randint(1, 2**31 - 1)


def now_ms() -> int:
    return int(time.time() * 1000)


def active_elements(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in data.get("elements", []) if not e.get("isDeleted")]


def bounds(els: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    if not els:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for e in els:
        x, y = float(e.get("x", 0)), float(e.get("y", 0))
        w, h = float(e.get("width", 0)), float(e.get("height", 0))
        xs += [x, x + w]
        ys += [y, y + h]
    return min(xs), min(ys), max(xs), max(ys)


def next_index(data: dict[str, Any], prefix: str = "z") -> str:
    """Allocate a simple unique index under prefix (load-safe enough for appends)."""
    existing = {e.get("index") for e in data.get("elements", [])}
    n = 0
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    while True:
        # z0, z1, ... z9, zA, ... then z10 style
        if n < len(chars):
            cand = f"{prefix}{chars[n]}"
        else:
            cand = f"{prefix}{n}"
        if cand not in existing:
            return cand
        n += 1


def base_el(**kwargs: Any) -> dict[str, Any]:
    el = {
        "id": nid(),
        "angle": 0,
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": seed(),
        "version": 1,
        "versionNonce": seed(),
        "isDeleted": False,
        "boundElements": None,
        "updated": now_ms(),
        "link": None,
        "locked": False,
    }
    el.update(kwargs)
    return el


def make_text(x: float, y: float, content: str, index: str, font_size: int = 20) -> dict[str, Any]:
    content = content.replace("\\n", "\n")
    lines = content.split("\n") or [""]
    width = max(8.0, max(len(line) for line in lines) * font_size * 0.6)
    height = max(font_size * 1.25, len(lines) * font_size * 1.25)
    return base_el(
        type="text",
        x=x,
        y=y,
        width=width,
        height=height,
        index=index,
        boundElements=[],
        text=content,
        fontSize=font_size,
        fontFamily=8,
        textAlign="left",
        verticalAlign="top",
        containerId=None,
        originalText=content,
        autoResize=True,
        lineHeight=1.25,
    )


def make_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    index: str,
    bg: str = "#fff8e7",
) -> dict[str, Any]:
    return base_el(
        type="rectangle",
        x=x,
        y=y,
        width=w,
        height=h,
        index=index,
        backgroundColor=bg,
        boundElements=[],
        roundness={"type": 3},
    )


def make_arrow(x1: float, y1: float, x2: float, y2: float, index: str) -> dict[str, Any]:
    dx, dy = x2 - x1, y2 - y1
    return base_el(
        type="arrow",
        x=x1,
        y=y1,
        width=max(abs(dx), 0.001),
        height=max(abs(dy), 0.001),
        index=index,
        roundness={"type": 2},
        points=[[0, 0], [dx, dy]],
        lastCommittedPoint=None,
        startBinding=None,
        endBinding=None,
        startArrowhead=None,
        endArrowhead="arrow",
        elbowed=False,
    )


def make_separator(x: float, y: float, width: float, index: str) -> dict[str, Any]:
    return base_el(
        type="line",
        x=x,
        y=y,
        width=width,
        height=0,
        index=index,
        strokeColor="#868e96",
        strokeWidth=1,
        strokeStyle="dashed",
        roughness=0,
        roundness={"type": 2},
        points=[[0, 0], [width, 0]],
        lastCommittedPoint=None,
        startBinding=None,
        endBinding=None,
        startArrowhead=None,
        endArrowhead=None,
    )


def take_index(data: dict[str, Any]) -> str:
    idx = next_index(data)
    # reserve by temporarily noting — caller appends element with this index
    return idx
