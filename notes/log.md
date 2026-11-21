# Session log

| Date | What I ran | Number | Notes |
|---|---|---|---|
| 2026-11-21 | `scripts/check_mps.py` | MPS=True; CPU ~169 ms; MPS ~123 ms (4096×4096) | Day 1 |
| 2026-11-21 | `src/embed.py` | hidden `(16, 10, 1024)`; fp32 batch16 **16.1 ms/sent**; fp16 batch8 **14.5 ms/sent** | Day 2 |
