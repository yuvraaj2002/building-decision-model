# Diagrams (Excalidraw only)

All architecture sketches live here as **`.excalidraw.json`**. No Mermaid / Markdown diagram files.

| File | What it shows |
|---|---|
| [01-biencoder-pipeline.excalidraw.json](01-biencoder-pipeline.excalidraw.json) | Dataset row → Qwen → cache → heads → scores |
| [02-head-architecture.excalidraw.json](02-head-architecture.excalidraw.json) | ProjectionHead with **bar height ∝ dim** (1024 tall, 512 shorter) + scorer |

Open with the Excalidraw editor extension (close/reopen the tab after agent edits).

Agent skill: `.cursor/skills/excalidraw-local` — edit via scripts, never by dumping raw JSON into chat.
