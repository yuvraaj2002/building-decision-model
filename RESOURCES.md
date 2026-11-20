# Resources

Everything I'm using to learn System One decision models and build my own. The rule: every resource has to feed into code I write. Items marked ⭐ are must-do.

---

## 1. Start here

Watch these with a goal, not as background noise. Pause at each chapter and try to rebuild that piece in a notebook before moving on.

| | Resource | Type | What to take from it |
|---|---|---|---|
| ⭐ | [I trained a Jev-like Decision Language Model from scratch](https://www.youtube.com/watch?v=sF3CNPbWA8o) (Neural Breakdown with AVB, 1:49:00) | Video | The **joint-encoder** route: Qwen3-0.6B plus a small decision head that scores choices without depending on their order. Closest to what I'll build first. |
| ⭐ | [Build a System One Model from Scratch (like Jev): Full Workshop](https://www.youtube.com/watch?v=vEpkc44bHVQ) (Vizuara, 1:51:58) | Video | The **bi-encoder / contrastive** route: a frozen Qwen3-8B, two small heads, a state · action dot product, hard negatives and 3-stage training (a rebuild of CLM-8B). |

**AVB video: key timestamps**
- `1:38` what Jev is · `3:43` what we're building · `8:08` choice / score / null (yes-no) questions
- `12:25` bi-encoders vs joint encoders · `14:26` the network on top of Qwen · `17:55` decision head
- `22:49` choice-order invariance (the trickiest part, watch twice) · `32:39` live coding in PyTorch
- `1:31:46` dataset · `1:41:35` training and results
- Goal: be able to explain why every choice gets the same starting position IDs and why the attention mask stops choices from seeing each other.
- The [original livestream](https://youtube.com/live/AzxoU7kxjig) has a longer architecture walkthrough if a part goes too fast.

**Vizuara workshop: key timestamps**
- `21:06` state and action · `25:00` two vectors and a dot product · `33:20` caching
- `48:10` replacing the output head · `55:20` pooling
- `1:05:20` data and loss · `1:09:55` hard negatives · `1:14:22` contrastive loss
- `1:20:40` three training stages · `1:33:36` runtime caching · `1:46:40` GPUs · `1:48:00` datasets
- Goal: be able to write the InfoNCE loss from memory and explain why caching action vectors makes this model cheap to run.

---

## 2. Core idea: what a System One model is

| | Resource | Type | What to take from it |
|---|---|---|---|
| ⭐ | [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (TypeSafe, 15 Sep 2026) | Launch post | The official framing: unstructured state in, typed probabilistic decisions out, outputs scored in parallel, calibrated confidence. Jev itself is closed. |
| ⭐ | [TypeSafe docs: Introduction](https://docs.typesafe.ai/introduction) | Docs | The API shape (state plus Choice / Score / Noul questions). Copy this interface for my own model. |
| ⭐ | [Contrastive Language Models (CLM)](https://contrastive-lm.notion.site) (Kwok et al., Stanford + NVIDIA, 23 Sep 2026) | Research blog | The open counterpart to Jev: a state encoder and an action encoder trained with InfoNCE, then used as a zero-shot action scorer. Covers the data recipe (pre / mid / post training) and scaling laws. |
| | [Jev and the rise of decision models](https://www.thoughtworks.com/en-ca/insights/blog/generative-ai/jev-decision-models-what-need-know-why) (Thoughtworks) | Article | Where decision models fit next to LLMs, and the three modes (noul, rating, choice) in plain words. |
| | [Jev Explained: Adding a Decision Layer Between LLMs and AI Agents](https://builder.aws.com/content/3JmcgnkwKYnhTCvMEsTOpcmV84k/jev-explained-adding-a-decision-layer-between-llms-and-ai-agents) (AWS Builder Center) | Article | The architecture view: the decision model gives a bounded judgment, and policy plus IAM decide what actually runs. Useful for the demo I build at the end. |

---

## 3. Concepts I need

### Bi-encoder vs joint (cross) encoder
| | Resource | Type | Why it matters for the build |
|---|---|---|---|
| ⭐ | [Sentence-BERT](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019) | Paper | The original case for encoding two texts separately and comparing vectors. That's the CLM design, and why action vectors can be cached. |
| | [SBERT docs: Cross-Encoders](https://sbert.net/examples/cross_encoder/applications/README.html) | Docs | When to use a cross-encoder (more accurate, no caching) vs a bi-encoder (fast). That's the AVB joint model vs CLM trade-off. |

### Contrastive learning / InfoNCE
| | Resource | Type | Why it matters for the build |
|---|---|---|---|
| ⭐ | [Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) (Lilian Weng) | Blog | The best single overview: InfoNCE, in-batch negatives, temperature and hard negatives in one place. Read before writing the loss. |
| ⭐ | [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021) | Paper | The symmetric contrastive loss with a learned temperature. CLM is basically CLIP with states and actions instead of images and captions. |
| | [SimCSE](https://arxiv.org/abs/2104.08821) (Gao et al., 2021) | Paper | Contrastive training for sentence embeddings with very little machinery. A good small-scale recipe to copy. |

### Hard negatives
| | Resource | Type | Why it matters for the build |
|---|---|---|---|
| ⭐ | [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906) (Karpukhin et al., 2020) | Paper | Shows in-batch negatives plus one BM25 hard negative per example make a big difference. Same idea as CLM's mid-training stage. |

### Pooling a decoder LLM into one vector
| | Resource | Type | Why it matters for the build |
|---|---|---|---|
| ⭐ | [Improving Text Embeddings with LLMs (E5-Mistral)](https://arxiv.org/abs/2401.00368) (Wang et al., 2023) | Paper | Last-token pooling on a decoder LLM plus contrastive fine-tuning with LoRA. The exact recipe for turning Qwen into a state/action encoder. |
| | [LLM2Vec](https://arxiv.org/abs/2404.05961) (BehnamGhader et al., 2024) | Paper | Mean vs last-token pooling, and making a causal model attend both ways. Good for pooling experiments. |
| | [NV-Embed](https://arxiv.org/abs/2405.17428) (Lee et al., 2024) | Paper | A latent-attention pooling layer and hard-negative training at scale. An upgrade path once the basic version works. |

### Scoring a list of options, order invariance
| | Resource | Type | Why it matters for the build |
|---|---|---|---|
| ⭐ | [Hugging Face: Multiple choice task guide](https://huggingface.co/docs/transformers/tasks/multiple_choice) | Docs | The simplest baseline: encode (context, option) pairs, one score each, softmax over options. Build this first, before the fancy versions. |
| | [Deep Sets](https://arxiv.org/abs/1703.06114) (Zaheer et al., 2017) | Paper | The theory of permutation invariance. Explains why the answer must not change when the options are shuffled. |

---

## 4. Code to read (and run)

| | Resource | Type | What to take from it |
|---|---|---|---|
| ⭐ | [avbiswas/bev-train](https://github.com/avbiswas/bev-train) | Repo | Start with `arch.ipynb`, then `tokenization.py` (shared position IDs, choice masks) and `network.py` (Qwen + LoRA + head). Run `configs/smoke.yaml` to overfit 10 questions first. |
| ⭐ | [VizuaraAI/system-one-from-scratch](https://github.com/VizuaraAI/system-one-from-scratch) | Repo | Full CLM rebuild: heads of 4096 → 1536 → 512, cosine times a learned scale, about 18.9M trainable params, the training stages and scripts. |
| ⭐ | [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) | Repo | The official serving engine, a TypeSafe-compatible API (Choice / Noul / Score) and a fine-tuning tutorial. Use its API as the target interface. |
| | [System One workshop site](https://system-one-workshop.vercel.app) | Slides / demos | Workshop slides, games and results. Good for revising after the video. |

---

## 5. Models and datasets

**Base models**
| | Resource | Type | What to take from it |
|---|---|---|---|
| ⭐ | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Model | Main backbone. Small enough for a free or cheap GPU, and it's what bev-train uses. |
| | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Model | The step up once the 0.6B pipeline works end to end. |
| | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Model | Already trained for embeddings. A strong starting point for the bi-encoder (CLM-style) version. |
| | [avbiswas/bev-decider-0.4B](https://huggingface.co/avbiswas/bev-decider-0.4B) | Model | The trained result of the AVB video (base Qwen3-0.6B). The baseline my model has to beat. CC-BY-NC. |

**Datasets**
| | Resource | Type | What to take from it |
|---|---|---|---|
| ⭐ | [avbiswas/bev-decision](https://huggingface.co/datasets/avbiswas/bev-decision) | Dataset | 100K–1M rows of choice, yes/no and score questions, including tool routing. The main training set. |
| | [neulab/adp-v2](https://huggingface.co/datasets/neulab/adp-v2) | Dataset | Agent trajectories, which Vizuara uses for post-training (state → action that was taken). |
| | [Contrastive-LM/CLM-v0.1-Pretrain-Nemotron](https://huggingface.co/datasets/Contrastive-LM/CLM-v0.1-Pretrain-Nemotron) | Dataset | CLM pre-training vectors (~1 TB). Don't download it. Read the card to see how the pairs are built. |
| | [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | Dataset | 60k queries with tool lists. Turn it into "pick the right tool" choice data. Gated: accept the terms on HF first. |
| | [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) | Dataset | Clean 5-option multiple choice. Good for checking the head and order invariance on a known task. |

---

## 6. Tools and compute

| | Resource | Type | What to take from it |
|---|---|---|---|
| ⭐ | [PEFT: LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora) | Docs | How to put LoRA on the Qwen backbone and train only the adapters plus the head. |
| | [LoRA paper](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) | Paper | What rank and alpha actually do, so I'm not tuning blind. |
| ⭐ | [Sentence Transformers: Training overview](https://sbert.net/docs/sentence_transformer/training_overview.html) | Docs | The fastest way to train the bi-encoder version: datasets, trainer and evaluators. |
| | [Sentence Transformers: Loss overview](https://sbert.net/docs/sentence_transformer/loss_overview.html) | Docs | `MultipleNegativesRankingLoss` is InfoNCE with in-batch negatives. Pick losses from here. |
| | [Sentence Transformers: util (`mine_hard_negatives`)](https://sbert.net/docs/package_reference/util/index.html) | Docs | Ready-made hard-negative mining for building the mid-training data. |
| | [Google Colab](https://colab.research.google.com/) | Compute | Free T4 for the smoke tests and the 0.6B LoRA runs. |
| | [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) | Compute | Free weekly GPU quota and longer sessions than Colab. |
| | [Modal](https://modal.com/pricing) | Compute | Pay-per-second GPUs. Vizuara ran its pipeline here, and it's the option for the 1.7B runs. |

---

## 7. Evaluation and further reading

| | Resource | Type | What to take from it |
|---|---|---|---|
| | [Terminal-Bench](https://www.tbench.ai/) | Benchmark | One of the agentic benchmarks CLM reports as a verifier (87.6% on 2.1). A stretch goal, not a first eval. |
| | [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) | Benchmark | Tool-selection eval ideas. Useful for testing a tool-routing decider. |
| | [jackyk02/qwen3_8b_head](https://huggingface.co/jackyk02/qwen3_8b_head) | Checkpoint | The public CLM head checkpoint Vizuara reverse-engineered the recipe from. Read its config for the real hyperparameters. |

My own evals before any benchmark: choice accuracy on a held-out split of bev-decision, the same accuracy with the options shuffled (it should not move), calibration (do 0.9-confidence answers come out right about 90% of the time?) and latency per decision.
