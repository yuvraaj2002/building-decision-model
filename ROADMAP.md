# Roadmap

Six weeks, one hour a day, building a System One decision model myself. Every session ends with code that runs and a number I measured. No session ends with "watched a video".

**My setup:** MacBook Air (Apple Silicon, 24 GB unified memory) for everything up to Qwen3-0.6B with a frozen encoder. Colab or Kaggle free T4 for anything heavier. Qwen3-8B and CLM-8B are things I read, not things I train.

---

## How I work

One session = one hour:

| Time | What |
|---|---|
| 10 min | Read: only the doc, paper section or video timestamp today's task needs |
| 40 min | Build: write the code, run it, get a number |
| 10 min | Log: one line in `notes/log.md` (date, what I ran, the number, what broke) and commit |

Rules:

- **Video rule:** at most one video segment a week, and only the timestamp the current milestone points to. If I want more, I write code instead.
- **No number, not done.** If a session doesn't produce a measurement, the next session finishes it before moving on.
- **Small first.** Every script runs on 20 rows before it runs on 2,000.
- **Cache everything.** Hidden states go to disk (`cache/*.pt` or `.safetensors`) once, and training reads from there. The Mac never re-encodes the same text twice.
- **Mac limits:** `device="mps"`, batch sizes small (start at 8), fp16 for the frozen forward pass where MPS handles it, fp32 fallback if I get NaNs. If a run needs more than about 16 GB or more than 40 minutes, it moves to Colab/Kaggle.
- **Weekends are buffer.** Catch up, or skip.

Repo layout I'll grow into:

```
src/        embed.py, pool.py, data.py, heads.py, train.py, eval.py, serve.py
notes/      log.md + one write-up per milestone
cache/      cached hidden states (gitignored)
results/    eval JSONs and plots
```

---

## Progress

| # | Milestone | Week | Sessions | Key number | Status |
|---|---|---|---|---|---|
| 0 | Setup and first embedding | 1 | 2 | ~16 ms/sent fp32 @ batch 16 on MPS | ☑ |
| 1 | Zero-shot baseline | 1 | 4 | top-1 **52.5%** labeled; p50 ~1.0s | ☑ |
| 2 | Bi-encoder heads + InfoNCE | 2–3 | 8 | accuracy vs baseline | ☐ |
| 3 | Hard negatives + order invariance | 3–4 | 6 | accuracy, shuffled accuracy | ☐ |
| 4 | Scale up + LoRA on a T4 | 5 | 5 | accuracy / cost / time per setup | ☐ |
| 5 | Ship it | 6 | 5 | p50 latency, calibration | ☐ |

Fill in as I go:

| Metric | Baseline (M1) | Heads (M2) | + Hard negs (M3) | + LoRA (M4) |
|---|---|---|---|---|
| Top-1 accuracy | **52.5%** (labeled) | ___ | ___ | ___ |
| Shuffled-option accuracy | ___ | ___ | ___ | ___ |
| Latency per decision (ms) | **p50 ~1035** (labeled, fp16, max_len 256) | ___ | ___ | ___ |

---

## Milestone 0: Setup and first embedding (week 1, 2 sessions)

**Goal:** a clean environment and one script that turns text into a vector on the Mac's GPU and tells me how long it took.

- [x] **Day 1:** `uv init`, add `torch transformers datasets accelerate`. Write `scripts/check_mps.py` that prints `torch.backends.mps.is_available()` and multiplies two 4096×4096 matrices on CPU vs MPS, timing both.
- [x] **Day 2:** `src/embed.py`: load `Qwen/Qwen3-0.6B` with `output_hidden_states=True`, embed 20 sentences, print the hidden-state shape, and time ms per sentence at batch sizes 1, 8 and 16 in fp32 vs fp16.

**Use:** [Qwen3-0.6B model card](RESOURCES.md#5-models-and-datasets) · [bev-train setup section](RESOURCES.md#4-code-to-read-and-run)

**Done when:** `uv run python src/embed.py` prints a shape and a ms/sentence number, and both are logged.

**Write it up:** `notes/00-setup.md`: "What running a 0.6B model on a MacBook Air actually costs, in milliseconds."

---

## Milestone 1: Zero-shot baseline (week 1, 4 sessions)

**Goal:** score a state against its options with no training at all (frozen encoder + pooling + cosine), so every later step has a number to beat.

- [x] **Day 3:** load `avbiswas/bev-decision`, print the columns and 5 rows, and write `src/data.py` that turns its choice/noul questions into `(state, question, choices, correct_idx)`. Save a fixed 200-row eval split to `data/eval_bev.jsonl` (seeded, never trained on). (`tau/commonsense_qa` deferred — bev is the primary state/context set.)
- [x] **Day 4:** write `pool()` with `mean` and `last_token` modes (masking padding properly). Check them on 20 pairs I write by hand (10 should match, 10 shouldn't) and print the cosine gap for each mode. Winner on gap: **mean** (0.179 vs last_token 0.140).
- [x] **Day 5:** `src/eval.py`: embed the state and each option, pick the argmax cosine, and report top-1 accuracy on the 200-row bev set, plus random-guess accuracy for comparison. Result: **47.5%** top-1 vs **39.4%** chance (+8.1); choice 32%, noul 65%.
- [x] **Day 6:** timing + prompt formats. Winner: **labeled** 52.5% (plain 47.5%); chance 39.4%; p50 ~1035 ms (fp16, max_length 256). Saved `results/baseline.json`.

**Use:** [Vizuara workshop 21:06 + 55:20](RESOURCES.md#1-start-here) (the one video segment this week) · [E5-Mistral and LLM2Vec, pooling sections](RESOURCES.md#pooling-a-decoder-llm-into-one-vector) · [HF multiple choice guide](RESOURCES.md#scoring-a-list-of-options-order-invariance)

**Done when:** `results/baseline.json` has accuracy (mean vs last-token), random-chance accuracy and p50 latency for both eval sets. An honest result is fine even if the baseline is close to random.

**Write it up:** `notes/01-baseline.md`: "A frozen LLM as a zero-shot decider: how far does cosine similarity get you?"

---

## Milestone 2: Bi-encoder heads + InfoNCE (weeks 2–3, 8 sessions)

**Goal:** the CLM idea at Mac scale: frozen Qwen3-0.6B, cached hidden states, two small trainable heads (state and action), trained with InfoNCE and in-batch negatives.

- [x] **Day 7:** `scripts/cache.py`: encode about 5k training rows once (states and options separately) with the winning pooling, and save to `cache/`. Log the encode time and the size on disk. Result: states **[5000, 1024]**, unique options **[4687, 1024]**, encode **1801 s** (~30 min), disk **20.32 MB**.
- [ ] **Day 8:** `src/heads.py`: two MLP heads (hidden dim → 512, LayerNorm, GELU), L2-normalised outputs, score = cosine × learned scale. Unit test: output shapes, and the score matrix is `[B, B]`.
- [ ] **Day 9:** write the InfoNCE loss myself (cross-entropy over the `[B, B]` similarity matrix, diagonal = positives). Check it against `MultipleNegativesRankingLoss` on random tensors and make sure the values match.
- [ ] **Day 10:** `src/train.py`: a training loop on cached vectors, MPS, batch 64. Overfit 32 pairs to about 0 loss first (sanity check).
- [ ] **Day 11:** train on the 5k cache, evaluate on the 200-row splits every epoch, and log train loss + eval accuracy to `results/m2_run1.json`.
- [ ] **Day 12:** sweep one thing: batch size (32 / 64 / 128) or temperature (fixed vs learned). One change per run.
- [ ] **Day 13:** grow the cache to about 20k rows (keep the size on disk in mind) and retrain the best setting.
- [ ] **Day 14:** read the matching parts of `system-one-from-scratch` (`train_heads.py`) and list 3 differences from my version. Try the one that's most likely to matter.

**Use:** [Vizuara workshop 25:00 + 1:05:20 + 1:14:22](RESOURCES.md#1-start-here) (one segment per week) · [Lilian Weng's contrastive post, CLIP](RESOURCES.md#contrastive-learning--infonce) · [Sentence-BERT](RESOURCES.md#bi-encoder-vs-joint-cross-encoder) · [system-one-from-scratch](RESOURCES.md#4-code-to-read-and-run) · [Sentence Transformers loss overview](RESOURCES.md#6-tools-and-compute)

**Done when:** trained heads beat the M1 baseline on the bev-decision eval split by at least ___ points (I set the target after seeing M1), and the training curve plus the eval numbers are in `results/`.

**Write it up:** `notes/02-infonce.md`: "InfoNCE in 20 lines: training a decision head on cached LLM vectors on a MacBook."

---

## Milestone 3: Hard negatives + order invariance (weeks 3–4, 6 sessions)

**Goal:** make the model pick between options that look alike, and prove the answer doesn't depend on option order.

- [ ] **Day 15:** shuffle test: run eval with options in their original order and with 5 random permutations. Report accuracy for each and the max difference. (A bi-encoder should be invariant by design, so if it isn't, I have a bug.)
- [ ] **Day 16:** mine hard negatives: for each state, take the top-k most similar wrong options from the cache using my M2 model (or `mine_hard_negatives`). Spot-check 20 by eye and log how many are actually wrong vs mislabeled.
- [ ] **Day 17:** add hard negatives to the loss (extra columns in the similarity matrix) and retrain.
- [ ] **Day 18:** ablation: in-batch only vs + 1 / 3 / 7 hard negatives. Accuracy and training time for each.
- [ ] **Day 19:** read bev-train's `tokenization.py` (shared position IDs, choice attention mask), then write a 30-line test showing why a *joint* encoder breaks order invariance without that trick.
- [ ] **Day 20:** error analysis: dump 30 wrong answers and sort them into buckets (label noise, too many options, long state, etc.).

**Use:** [AVB video 12:25 + 22:49](RESOURCES.md#1-start-here) · [DPR](RESOURCES.md#hard-negatives) · [Deep Sets](RESOURCES.md#scoring-a-list-of-options-order-invariance) · [bev-train](RESOURCES.md#4-code-to-read-and-run) · [Sentence Transformers util (](RESOURCES.md#6-tools-and-compute)`mine_hard_negatives`[)](RESOURCES.md#6-tools-and-compute)

**Done when:** hard negatives add at least ___ points over M2, and shuffled-option accuracy equals normal accuracy (difference of 0 on the bi-encoder, logged across 5 permutations).

**Write it up:** `notes/03-hard-negatives.md`: "Easy negatives teach nothing: what hard negatives did to my decision model, and a shuffle test every decider should pass."

---

## Milestone 4: Scale up + LoRA on a T4 (week 5, 5 sessions)

**Goal:** find out whether training the encoder itself (LoRA) is worth it compared with frozen + heads, using free GPUs.

- [ ] **Day 21:** port the pipeline to a Colab/Kaggle notebook (`pip install`, clone the repo, same eval split). Re-run M3's best config on the T4 and confirm the numbers match the Mac.
- [ ] **Day 22:** add LoRA (PEFT) to Qwen3-0.6B's attention layers, train heads + LoRA end to end on a small subset (no cache possible now), and log GPU memory and step time.
- [ ] **Day 23:** a full LoRA run within one free session. Log accuracy, wall-clock time and GPU-hours.
- [ ] **Day 24:** optional: try `Qwen3-Embedding-0.6B` as the frozen encoder with my heads (starting from embedding training instead of a chat model).
- [ ] **Day 25:** comparison table: frozen+heads vs LoRA vs Embedding-0.6B. Accuracy, train time, GPU-hours, size on disk, and whether action vectors can still be cached. Pick the one to ship.

**Use:** [PEFT LoRA guide + LoRA paper](RESOURCES.md#6-tools-and-compute) · [Qwen3-Embedding-0.6B, bev-decider-0.4B](RESOURCES.md#5-models-and-datasets) · [Colab / Kaggle](RESOURCES.md#6-tools-and-compute) · [Vizuara 1:20:40 three stages](RESOURCES.md#1-start-here)

**Done when:** `results/m4_comparison.md` has the table, and I can say in one sentence which setup wins on accuracy per GPU-hour. Optionally I also compare against `bev-decider-0.4B` on the same split.

**Write it up:** `notes/04-lora-vs-frozen.md`: "Should you fine-tune the encoder? Frozen heads vs LoRA on a free T4, with real numbers."

---

## Milestone 5: Ship it (week 6, 5 sessions)

**Goal:** a usable decider behind an API, with an honest eval report.

- [ ] **Day 26:** `src/serve.py`: a FastAPI `POST /decide` taking `{state, options}` and returning `{choice, probs}` (softmax over scores). Option vectors are cached by hash, so repeated options skip the encoder.
- [ ] **Day 27:** load test: 200 requests, p50 / p95 latency with a cold vs warm option cache.
- [ ] **Day 28:** LLM baseline: the same 200 eval rows through Qwen3-0.6B *generating* the answer (prompt with lettered options, parse the letter). Compare accuracy, p50 latency and parse failures against `/decide`.
- [ ] **Day 29:** calibration: bucket predictions by confidence (0.5–0.6 … 0.9–1.0), compare accuracy per bucket and compute ECE. Plot to `results/calibration.png`.
- [ ] **Day 30:** `results/REPORT.md` (accuracy, shuffled accuracy, calibration, latency vs the LLM baseline, known failure modes), then push the heads to Hugging Face with a model card (base model, data, eval numbers, limits, license notes).

**Use:** [CLM repo: its API (Choice / Noul / Score) is the target interface](RESOURCES.md#4-code-to-read-and-run) · [TypeSafe docs](RESOURCES.md#2-core-idea-what-a-system-one-model-is) · [Vizuara 1:33:36 runtime caching](RESOURCES.md#1-start-here) · [Evaluation section](RESOURCES.md#7-evaluation-and-further-reading)

**Done when:** `/decide` runs locally, the p50 latency is measured next to the generation baseline, and REPORT.md plus the HF model card are published.

**Write it up:** `notes/05-ship.md`: "I built a tiny System One model: one forward pass, cached options, ___ ms per decision. Here's the full report."

---

## After week 6 (only if the above is done)

- Add `noul` (yes/no) and `score` question types, the way bev-train does.
- Try the joint-encoder version from the AVB video and compare it against my bi-encoder on the same eval.
- Use the decider as a tool router for one of my own agents and measure how often it picks the right tool.
