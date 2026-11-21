# Load Qwen/Qwen3-0.6B on MPS.
# output_hidden_states belongs on the forward call, not from_pretrained.
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to(device)
model.eval()

# 20 short sample sentences (different from each other)
sample_sentences = [
    "The kitchen timer beeped twice.",
    "Rain tapped softly on the window.",
    "She booked a train for Tuesday morning.",
    "The model scored every option in one pass.",
    "Coffee cooled beside an open notebook.",
    "He chose the quieter route home.",
    "A stray cat watched from the fence.",
    "The API returned a calibrated confidence score.",
    "Leaves spun across the empty courtyard.",
    "They postponed the meeting until Friday.",
    "Hidden states were cached to disk once.",
    "The bakery sold out of sourdough early.",
    "Fog blurred the harbor lights at dawn.",
    "Batch size sixteen fit in memory.",
    "Someone left a map on the cafe table.",
    "The bike chain needed oil again.",
    "Cosine similarity picked the wrong option.",
    "Thunder rolled past the distant hills.",
    "She labeled the eval split with seed forty-two.",
    "Two sparrows argued over a crumb.",
]

batch_sizes = [1, 8, 16]
precisions = ["fp32", "fp16"]


def sync():
    if device == "mps":
        torch.mps.synchronize()


def run_inference(batch_size: int, precision: str):
    """Tokenize `batch_size` sentences, run one forward, return shape + timings."""
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model.to(dtype=dtype)

    # Pad to a real batch tensor: [B, seq_len] — lists of 1-row tensors won't work
    texts = sample_sentences[:batch_size]
    batch = tokenizer(texts, return_tensors="pt", padding=True)
    # Moving all the tensors to the device on which model exists
    batch = {k: v.to(device) for k, v in batch.items()}

    # Running the model in inference mode
    with torch.no_grad():
        # Warmup (skip first MPS kernel compile in the timed number)
        _ = model(**batch, output_hidden_states=True)
        sync()

        sync()
        start = time.monotonic()
        out = model(**batch, output_hidden_states=True)
        sync()
        elapsed_ms = (time.monotonic() - start) * 1000

    hidden = out.hidden_states[-1]  # [B, seq_len, hidden_size]
    return hidden.shape, elapsed_ms, elapsed_ms / batch_size


print(f"device={device}  hidden_size={model.config.hidden_size}")
print(f"{'batch':>6}  {'prec':>5}  {'hidden_shape':>22}  {'total_ms':>10}  {'ms/sent':>10}")

for precision in precisions:
    for batch_size in batch_sizes:
        shape, total_ms, ms_per_sent = run_inference(batch_size, precision)
        print(
            f"{batch_size:>6}  {precision:>5}  {str(tuple(shape)):>22}  "
            f"{total_ms:>10.2f}  {ms_per_sent:>10.2f}"
        )

model.to(dtype=torch.float32)
