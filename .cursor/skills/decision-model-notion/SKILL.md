---
name: decision-model-notion
description: >-
  Log Decision Model build sessions and progress to the pinned Notion page under
  Work. Use when ending a session, logging numbers, updating Day/Milestone
  checkboxes, writing notes for this project, or when the user mentions Notion
  notes for Decision Model / building-decision-model.
---

# Decision Model Notion Notes

## Pinned destinations (do not search)

Use these IDs/URLs directly. **Never** run `notion-search` to find this page unless the user says it moved or the fetch fails.

| Entity | Value |
|---|---|
| **Page title** | Decision Model |
| **Page ID** | `41f6d344-dcd6-813a-b368-cdaf07ff42fd` |
| **Page URL** | https://app.notion.com/p/41f6d344dcd6813ab368cdaf07ff42fd |
| **Parent** | Work |
| **Parent page ID** | `4146d344-dcd6-81b7-8c0b-f641249f28e8` |
| **Parent URL** | https://app.notion.com/p/4146d344dcd681b78c0bf641249f28e8 |
| **Icon** | 🧠 |

MCP namespace: `plugin-notion-workspace-notion`.

## When to update

Update the Notion page when any of these happen:

1. A Day / session finishes (or the user asks to log)
2. A measurement is produced (accuracy, ms, shape, loss, etc.)
3. A Progress checkbox should flip
4. A milestone write-up belongs in **Notes**

Also mirror one line into local `notes/log.md` when that file exists (roadmap rule). Notion is the durable session notebook; the repo log stays short.

## Page sections (keep this structure)

1. Top callout — project one-liner (leave alone unless framing changes)
2. **Overview** — repo / local path / cadence
3. **Progress** — checkboxes for Day 1, Day 2, then Milestones 1–5
4. **Session log** — newest day entries; chronological under this heading
5. **Notes** — longer write-ups (`notes/00-setup.md` style content)

Do not rename these headings. Anchors and edits depend on them.

## Workflow (every log)

1. **Fetch once** with the pinned URL/ID (`notion-fetch`). Do not search.
2. Edit with `notion-update-page` on `page_id` = pinned ID or URL.
3. Prefer surgical edits:
   - `command: "update_content"` + `content_updates` for checkbox flips and rewriting an existing day block
   - `command: "insert_content"` to add a new `### Day N — YYYY-MM-DD` block under **Session log**
4. Avoid `replace_content` unless restructuring the whole page (preserves nothing by accident).
5. Confirm back to the user with the page URL.

### Insert a new day entry

Use `insert_content` with `position` after the **Session log** heading (or after the latest day heading if that is clearer from the fetch). Template:

```markdown
### Day N — YYYY-MM-DD

- Ran: `path/to/script`
- Number: **…** (shape / ms / accuracy / loss)
- Broke: … (or "nothing")
- Status: done / blocked / continue next session
```

Rules from the roadmap: **no number, not done**. Always include the measured number.

### Flip Progress checkboxes

Match the exact checkbox line from the fetch (including backticks/code spans), then replace `- [ ]` → `- [x]` for completed days/milestones only.

### Longer Notes

Append under **Notes** with a short heading, e.g. `### 00 — Setup` or `### 01 — Baseline`. Keep Milestone write-ups here when the user asks; otherwise session bullets stay in **Session log**.

## Auth / failures

If Notion tools return `needsAuth` or auth errors: run `mcp_auth` for `plugin-notion-workspace-notion`, ask the user to approve, then retry with the **pinned ID** (still no search).

If fetch by ID fails with not-found: tell the user the page may have moved; only then search under parent Work (`page_id` `4146d344-dcd6-81b7-8c0b-f641249f28e8`) and update this skill with the new ID.

## Do not

- Create a second "Decision Model" page under Work
- Dump full chat transcripts into Notion
- Log sessions that produced no measurement (finish the measurement first)
- Put secrets, tokens, or HF keys on the page
