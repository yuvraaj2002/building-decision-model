# Session log

| Date | What I ran | Number | Notes |
|---|---|---|---|
| 2026-11-21 | `scripts/check_mps.py` | MPS=True; CPU ~169 ms; MPS ~123 ms (4096×4096) | Day 1 |
| 2026-11-21 | `src/embed.py` | hidden `(16, 10, 1024)`; fp32 batch16 **16.1 ms/sent**; fp16 batch8 **14.5 ms/sent** | Day 2 |
| 2026-11-21 | `src/data.py` | 202,727 converted (choice+noul); **200** seeded eval → `data/eval_bev.jsonl` | Day 3; CSQA deferred |
| 2026-11-21 | `scripts/pool_check.py` | mean gap **0.179**; last_token gap **0.140** → use **mean** | Day 4 |
| 2026-11-21 | `src/eval.py --save` | top-1 **47.5%** (95/200); chance **39.4%**; lift **+8.1** | Day 5; choice 32%, noul 65% |
| 2026-11-21 | `src/eval.py --compare-formats --save` | labeled **52.5%** vs plain **47.5%**; p50 ~**1035** / **970** ms | Day 6; fp16 max_len 256; M1 done |
| 2026-11-22 | `scripts/cache.py --n 5000` | states **[5000, 1024]**; unique options **[4687, 1024]**; encode **1801 s** (~30 min); disk **20.32 MB** | Day 7; labeled/mean/fp16/max_len 256; eval held out |
| 2026-11-23 | `src/heads.py` | outs **[4, 512]**; score_matrix **[4, 4]**; L2≈1; scale 10; **1,576,961** params | Day 8; ProjectionHead + BiEncoderHeads; diagrams/ + excalidraw-local skill |
