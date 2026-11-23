---
name: excalidraw-local
description: >-
  Create and refine local Excalidraw diagrams (.excalidraw / .excalidraw.json)
  in the repo via token-efficient scripts. Use when the user asks to draw, update,
  refine, or append flows in Excalidraw, or when editing diagrams/*.excalidraw*.
  Prefer this over reading or rewriting the full JSON in chat.
---

# Local Excalidraw (token-efficient)

Edit diagrams with **scripts**. Do **not** `Read` the full `.excalidraw.json` into context (files often embed multi‑MB PNGs and will break the editor if hand-edited badly).

## Hard rules

1. **Never dump the whole file into the model.** Use `summarize.py` for understanding.
2. **Mutate via scripts** (`new.py`, `append_flow.py`, `append_nn_stack.py`, `add_note.py`), then `validate.py`.
3. After edits: tell the user to **close and reopen** the Excalidraw tab if it was already open.
4. Keep diagrams under a project folder like `diagrams/` and commit the `.excalidraw.json` (repo-local).
5. Do not set image `status` to `pending`. Leave embedded files alone unless `validate.py` reports them.
6. **Decision Model:** architecture diagrams are **Excalidraw only** (`.excalidraw.json`). Do not add Mermaid / Markdown diagram files.

## Scripts

All paths below are relative to this skill’s `scripts/` directory. Run with the workspace file path as the first argument.

### Summarize (default first step)

```bash
python3 scripts/summarize.py path/to/file.excalidraw.json
```

Prints: element counts, canvas bounds, text labels, image statuses. **Use this instead of reading the file.**

### Validate

```bash
python3 scripts/validate.py path/to/file.excalidraw.json
```

Exits non-zero on duplicate ids, missing fields, or `image` with `status=pending`.

### Create empty diagram

```bash
python3 scripts/new.py path/to/file.excalidraw.json
```

### Append a horizontal flow below existing content

```bash
python3 scripts/append_flow.py path/to/file.excalidraw.json \
  --title "Refined v0 — …" \
  --steps "Step A|Step B|Step C" \
  --note "Optional principle note" \
  --detail "Optional detail box text"
```

- Steps are `|`-separated; use `\n` inside a step for line breaks.
- Places content **below** current bounds with a dashed separator.
- Arrows are **unbound** (load-safe).

### Append a neural-net dimension stack (tall/short bars)

```bash
python3 scripts/append_nn_stack.py path/to/file.excalidraw.json \
  --title "ProjectionHead" \
  --layers "cached:1024|Linear:512|LN:512|GELU:512|Linear:512|L2:512" \
  --note "Bar height ∝ dim — 1024 tall, 512 shorter"
```

Each layer is `label:dim`. Bar height scales with dim so vector width is visible.

### Add a free note below

```bash
python3 scripts/add_note.py path/to/file.excalidraw.json --text "…"
```

## Workflow

**Create**
1. Agree on title + step labels with the user (short).
2. `new.py` if file missing.
3. `append_flow.py` with steps.
4. `validate.py`.
5. User opens/reopens in Excalidraw editor.

**Refine**
1. `summarize.py` only.
2. Map the user’s change to one script call (new flow row, note, or replace via new file section).
3. Prefer **append a new section below** over rewriting earlier shapes (cheaper, safer).
4. `validate.py`.

**If load fails (“Unable to load initial data”)**
1. `validate.py` — fix `pending` images (`--fix-pending-images`).
2. If still broken, `summarize.py` then recreate refined section into a sibling `*_refined.excalidraw.json` with `new.py` + `append_flow.py`.

## Style defaults

- Stroke `#1e1e1e`, flow boxes `#fff8e7`, detail boxes `#e8f4ff`
- Font family `8` (Excalidraw default in this editor)
- Roughness `1` for shapes; separator line dashed grey

## Anti-patterns

- Reading/writing the raw JSON with the Read/Write tools
- Hand-crafting `boundElements` / arrow bindings in chat
- Inlining base64 images into prompts
- Replacing the whole document to change one label (append or targeted script instead)
