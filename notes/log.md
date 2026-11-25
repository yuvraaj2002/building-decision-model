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
| 2026-11-24 | `src/loss.py` | B=16 random **2.823737** (log 16 = 2.7726); perfect **0.005906**; MNRL diff **0.00e+00** | Day 9; InfoNCE (state→action + symmetric option) verified vs `MultipleNegativesRankingLoss` |
| 2026-11-24 | `src/train.py --overfit 32` | loss **3.5073 → 0.00616**; in-batch top1 **3.1% → 100.0%**; **3.61 s** on MPS | Day 10; 200 steps, AdamW lr=1e-3, scale 9.99→10.94; sanity check passed |
| 2026-11-24 | `src/train.py --epochs 10 --batch-size 64` | train loss **2.9772 → 1.6468**; best eval top1 **52.5%** (ep 7; choice 38.7%, noul 68.1%) vs 52.5% M1; **7.96 s** | Day 11; eval cached in 83.2s; saved `results/m2_run1.json`; likely hurt by duplicate yes/no in-batch false negatives |
