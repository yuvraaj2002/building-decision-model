# Decision Model: 10-Post Blog Series (DRAFT PLAN)

**Status:** draft for refinement. Headings, subheadings, and questions are proposals. Change anything.

**How each post gets written**
1. **Me first:** answer the "Your understanding" questions in your own words, in `blog/drafts/NN-slug.md`. No AI yet.
2. **Diagrams:** sketch the listed diagrams in Excalidraw (`diagrams/`) or as simple images in `blog/images/`.
3. **AI drafts:** I turn your answers plus the result files into a draft. Every number must trace to a file in `results/` or `notes/log.md`.
4. **You edit** into your own voice, then publish.

**Source of truth for numbers:** `results/*.json`, `notes/log.md`, `ROADMAP.md`.
**Concept sources:** [RESOURCES.md](../RESOURCES.md).

---

## Series at a glance

| # | Working title | Roadmap milestone | Week | Status |
|---|---|---|---|---|
| 1 | A frozen LLM as a zero-shot decider | M1 Zero-shot baseline | 1 | Numbers ready |
| 2 | Heads on top: pooling and encoder choices | M2 (part 1) | 2 | Numbers ready |
| 3 | InfoNCE from scratch on cached vectors | M2 (part 2) | 2–3 | Numbers ready |
| 4 | Why my trained heads tied the baseline | M2 (results) | 3 | Numbers ready |
| 5 | Hard negatives: easy negatives teach nothing | M3 (part 1) | 3–4 | Waiting on Days 16–18 |
| 6 | The shuffle test: order invariance | M3 (part 2) | 4 | Waiting on Day 15 |
| 7 | Caching everything: what the Mac can afford | M0 + M2 Day 7 | 1 / 2 | Numbers ready |
| 8 | Frozen heads vs LoRA on a free T4 | M4 | 5 | Waiting on M4 |
| 9 | Can the decider say how sure it is? | M5 (calibration) | 6 | Waiting on M5 |
| 10 | Shipping the decider and an honest report | M5 (serve + report) | 6 | Waiting on M5 |

---

## Post 1: A frozen LLM as a zero-shot decider

**Milestone:** M1, Week 1. **Hook:** A model that was never trained for this task still beats random chance, and here is how much.

### Subheadings
1. The task: one state, a list of options, pick one, no text generation
2. Turning text into one vector: mean pooling over Qwen3-0.6B hidden states
3. Cosine similarity as the score, and argmax as the decision
4. The chance baseline: 39.4%
5. Prompt format: labeled vs plain (52.5% vs 47.5%)
6. What 52.5% actually means on 200 rows

### Your understanding (answer before AI drafts)
- What exactly is a "state" and what is an "option" in the bev-decision data?
- Why can a frozen model score above chance with no training at all? What is the model's hidden state actually encoding?
- Why does mean pooling beat last-token pooling for this? (0.179 vs 0.140 gap)
- What does cosine ignore that a learned score would not?
- The labeled-vs-plain gap is 5 points on 200 rows. Is that a real difference? Why or why not?

### Diagrams
- Pipeline: state and options → Qwen3 → mean pool → cosine → argmax (adapt [01-biencoder-pipeline](../diagrams/01-biencoder-pipeline.excalidraw.json))
- Bar chart: chance (39.4%) vs plain (47.5%) vs labeled (52.5%)

### Numbers to cite
- `results/baseline.json`: top-1, chance, p50 latency (~1035 ms labeled, ~970 ms plain)
- Pooling gap: `notes/log.md`, Day 4

### Sources
- [Pooling a decoder LLM into one vector](../RESOURCES.md#pooling-a-decoder-llm-into-one-vector)
- [Scoring a list of options](../RESOURCES.md#scoring-a-list-of-options-order-invariance)

---

## Post 2: Heads on top: pooling and encoder choices

**Milestone:** M2, part 1, Week 2. **Hook:** The encoder is fixed. Everything that can be learned sits in two small heads on top.

### Subheadings
1. Why freeze the encoder: the Mac budget and the cache rule
2. Pooling recap: what mean vs last-token change in the vector
3. The two-tower idea: a state head and an action head
4. ProjectionHead: 1024 → 512, LayerNorm, GELU
5. L2 normalization, and why cosine becomes a dot product
6. The learned scale: what it does to the softmax
7. Encoder choice preview: Qwen3-0.6B chat model vs Qwen3-Embedding-0.6B (saved for M4)

### Your understanding (answer before AI drafts)
- Why two heads instead of one shared head? What breaks if they share weights?
- What does L2 normalization do to a dot product, and why does that matter for cosine?
- What does the learned scale control? What happens to the loss if it is too small?
- Why does pooling choice matter more when the encoder is frozen?

### Diagrams
- ProjectionHead architecture with bar heights proportional to dimension (existing: [02-head-architecture](../diagrams/02-head-architecture.excalidraw.json))
- Two towers feeding one [B, B] score matrix

### Numbers to cite
- Outputs `[4, 512]`, score matrix `[4, 4]`, L2 ≈ 1, scale 10, 1,576,961 params (`notes/log.md`, Day 8)

### Sources
- [Bi-encoder vs joint (cross) encoder](../RESOURCES.md#bi-encoder-vs-joint-cross-encoder)
- [Pooling a decoder LLM into one vector](../RESOURCES.md#pooling-a-decoder-llm-into-one-vector)

---

## Post 3: InfoNCE from scratch on cached vectors

**Milestone:** M2, part 2, Weeks 2–3. **Hook:** The loss is one cross-entropy over a matrix, and in-batch negatives are free.

### Subheadings
1. The contrastive idea: pull matched pairs together, push the rest apart
2. In-batch negatives: every other option in the batch is a negative
3. The [B, B] similarity matrix, with the diagonal as positives
4. Why random loss starts near log(B)
5. Symmetric loss: state→option and option→state
6. Checking my loss against MultipleNegativesRankingLoss
7. The overfit-32 sanity check

### Your understanding (answer before AI drafts)
- Why is the loss at random initialization close to log(16) = 2.77?
- What is a false negative in in-batch training? Can you give one from your own data?
- What does the temperature (scale) change in the softmax?
- Why did I check against MNRL, and what does an exact match (diff 0.00e+00) prove?

### Diagrams
- New: InfoNCE similarity matrix heatmap, diagonal highlighted, with an example false negative circled
- New: loss curve for the overfit-32 run (3.5073 → 0.00616)

### Numbers to cite
- Random B=16: 2.823737 (log 16 = 2.7726); perfect: 0.005906; MNRL diff: 0.00e+00
- Overfit-32: 3.5073 → 0.00616, in-batch top-1 3.1% → 100%, 3.61 s (`notes/log.md`, Days 9–10)

### Sources
- [Contrastive learning / InfoNCE](../RESOURCES.md#contrastive-learning--infonce)

---

## Post 4: Why my trained heads tied the baseline

**Milestone:** M2 results, Week 3. **Hook:** Training did not beat the zero-shot model. That result is the most useful thing I have so far.

### Subheadings
1. The result: 52.5% trained, 52.5% zero-shot, and why a tie is suspicious
2. Per-type breakdown: choice 38.7% vs noul 68.1%
3. The training curve: eval dipped early, then recovered by epoch 7
4. The hypothesis: duplicate yes/no options as in-batch false negatives
5. How I would test it (the Day 12–13 sweep)
6. What a negative really means in this dataset

### Your understanding (answer before AI drafts)
- In this dataset, can the same option text appear twice in one batch? Where does that come from?
- If two identical options sit on the diagonal and off-diagonal, what does the loss tell the model to do?
- Why would the choice type score so much lower than noul?
- Is a tie with the baseline a failure, or a finding? Argue both sides.

### Diagrams
- Training curve: train loss and eval top-1 by epoch (plot from `results/m2_run1.json`, save to `blog/images/`)
- Per-type bar chart: choice vs noul vs overall

### Numbers to cite
- `results/m2_run1.json`: best epoch 7, eval top-1 0.525, per-epoch history
- Sweep results once Day 12 is done

### Sources
- [Bi-encoder vs joint (cross) encoder](../RESOURCES.md#bi-encoder-vs-joint-cross-encoder)

---

## Post 5: Hard negatives: easy negatives teach nothing

**Milestone:** M3, part 1, Weeks 3–4. **Hook:** Random negatives are too easy to teach the model anything. Hard ones do, but they can also be wrong.

### Subheadings
1. Easy vs hard negatives, with examples from my data
2. Mining hard negatives with the M2 model
3. Spot-checking 20 by hand: how many are actually wrong?
4. Adding extra columns to the similarity matrix
5. Ablation: in-batch only vs +1 / +3 / +7 hard negatives
6. Accuracy vs training time

### Your understanding (answer before AI drafts)
- What makes a negative "hard" in terms of the model's score?
- When is a hard negative actually a correct answer? How would you spot that?
- What does adding columns do to the loss, and to training time?
- What does the ablation curve tell you about where to stop?

### Diagrams
- New: similarity matrix with extra hard-negative columns beside the in-batch block
- New: ablation chart, accuracy vs number of hard negatives

### Numbers to cite
- Pending: M3 Days 16–18 (`results/`, `notes/log.md`)

### Sources
- [Hard negatives](../RESOURCES.md#hard-negatives)

---

## Post 6: The shuffle test: order invariance

**Milestone:** M3, part 2, Week 4. **Hook:** A decider should give the same answer no matter how the options are listed. Most models don't pass this by default.

### Subheadings
1. Why option order should not matter
2. Why a bi-encoder is invariant by construction
3. Why a joint encoder is not, and the shared-position-ID trick
4. The shuffle test protocol: original order plus 5 random permutations
5. What a non-zero gap means: a bug, not a model property

### Your understanding (answer before AI drafts)
- Why is a bi-encoder invariant to option order by design? Trace it through the code path.
- What would change in a joint encoder if the options were reordered?
- If the shuffle gap is not zero on the bi-encoder, where do you look first?

### Diagrams
- New: same options in two orders, same scores, side by side
- New: joint encoder attention with and without shared position IDs (sketch)

### Numbers to cite
- Pending: M3 Day 15 (accuracy per permutation, max difference)

### Sources
- [Scoring a list of options, order invariance](../RESOURCES.md#scoring-a-list-of-options-order-invariance)
- [Bi-encoder vs joint (cross) encoder](../RESOURCES.md#bi-encoder-vs-joint-cross-encoder)

---

## Post 7: Caching everything: what the Mac can afford

**Milestones:** M0 setup and M2 Day 7, Weeks 1–2. **Hook:** Encoding is the expensive part. Encode once, store it, and never pay twice.

### Subheadings
1. What running a 0.6B model on a MacBook Air costs per sentence (fp32 vs fp16, batch sizes)
2. Encode once: 5,000 states took 1801 s (~30 min)
3. Deduplicating options: 5,000 states but only 4,687 unique options
4. Disk size: 20.32 MB for the cache
5. The trade-off: a frozen encoder means no fine-tuning of the encoder

### Your understanding (answer before AI drafts)
- Why does batch size change ms per sentence? What is the limit?
- Why does deduplicating options save work, and what would break if two options had the same text but different meaning?
- What exactly can't you do once the encoder is frozen and cached?

### Diagrams
- [01-biencoder-pipeline](../diagrams/01-biencoder-pipeline.excalidraw.json) with the cache step highlighted
- Bar chart: ms per sentence by batch size and precision

### Numbers to cite
- fp32 batch 16: 16.1 ms/sent; fp16 batch 8: 14.5 ms/sent (`notes/log.md`, Day 2)
- Cache: states [5000, 1024], unique options [4687, 1024], 1801 s, 20.32 MB (Day 7)

### Sources
- [Models and datasets](../RESOURCES.md#5-models-and-datasets)

---

## Post 8: Frozen heads vs LoRA on a free T4

**Milestone:** M4, Week 5. **Hook:** Is it worth training the encoder itself? I compared the options with real numbers and free GPUs.

### Subheadings
1. What LoRA changes: low-rank adapters on the attention layers
2. Why caching stops working once the encoder trains
3. Moving the pipeline to Colab/Kaggle and checking the numbers match the Mac
4. GPU memory and step time on a T4
5. The comparison: frozen+heads vs LoRA vs Qwen3-Embedding-0.6B
6. Accuracy per GPU-hour: which one to ship

### Your understanding (answer before AI drafts)
- What does a low-rank update save in parameters and memory? Why does it work at all?
- What do you lose by training end to end compared with caching?
- For your own project, when is frozen enough?

### Diagrams
- New: frozen vs LoRA training graph (what has gradients, what doesn't)
- New: comparison table as a chart (accuracy vs GPU-hours)

### Numbers to cite
- Pending: M4 comparison (`results/m4_comparison.md`)

### Sources
- [Tools and compute](../RESOURCES.md#6-tools-and-compute)
- [Models and datasets](../RESOURCES.md#5-models-and-datasets)

---

## Post 9: Can the decider say how sure it is?

**Milestone:** M5, Week 6. **Hook:** A decider that gives a probability is only useful if the probability means something.

### Subheadings
1. From scores to probabilities: softmax over the scale-adjusted scores
2. Confidence buckets: 0.5–0.6 up to 0.9–1.0
3. Expected calibration error (ECE), defined step by step
4. The reliability diagram
5. What overconfidence looks like for a decider, and what I'd do about it

### Your understanding (answer before AI drafts)
- If the model says 0.9 confidence, what should that mean? How would you check?
- Why is a softmax over cosine scores not automatically a calibrated probability?
- What can ECE hide, even when it looks low?

### Diagrams
- Reliability diagram: accuracy vs confidence, with the diagonal (`results/calibration.png`)
- Histogram of confidence values on the 200-row eval set

### Numbers to cite
- Pending: M5 Day 29 (per-bucket accuracy, ECE)

### Sources
- [Evaluation and further reading](../RESOURCES.md#7-evaluation-and-further-reading)

---

## Post 10: Shipping the decider, and an honest report

**Milestone:** M5, Week 6. **Hook:** A forward pass with cached options against an LLM that generates an answer. Which one is actually better, and by how much?

### Subheadings
1. The `/decide` API: state and options in, choice and probabilities out
2. Caching option vectors by hash: cold vs warm latency
3. Load test: 200 requests, p50 and p95
4. The generation baseline: Qwen3-0.6B picks a lettered option, parse failures included
5. REPORT.md: accuracy, shuffled accuracy, calibration, latency, failure modes
6. Publishing the heads to Hugging Face with a model card

### Your understanding (answer before AI drafts)
- Why is one forward pass with cached vectors cheaper than generating tokens?
- What does the warm option cache help with, and what does it not help with?
- Is the latency comparison fair? What would make it apples-to-apples?
- What should go in the failure-modes section, from what you have actually seen?

### Diagrams
- New: `/decide` request flow with cache hit and cache miss paths
- New: latency bars: decider cold, decider warm, generation baseline

### Numbers to cite
- Pending: M5 Days 26–29 (`results/REPORT.md`, load test, calibration)

### Sources
- [CLM repo: target interface](../RESOURCES.md#4-code-to-read-and-run)
- [Evaluation and further reading](../RESOURCES.md#7-evaluation-and-further-reading)

---

## Refinement questions (for us to decide together)

1. **Order:** Post 7 (caching and the M0 numbers) is chronologically early but sits late in the series. Move it to post 2, or keep it as a systems interlude?
2. **Length:** Is each post aimed at about 1,000 words, or shorter?
3. **Platform:** Where will these be published (personal site, Medium, Notion, GitHub)? This decides the image format and the front matter.
4. **Diagrams:** Keep everything in Excalidraw, or use plots from matplotlib for the charts?
5. **Pace:** One post per week while I code, or batch two posts per weekend?
