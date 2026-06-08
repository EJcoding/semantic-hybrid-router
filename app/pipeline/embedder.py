"""
app/pipeline/embedder.py
========================
Phase 2: Deep Learning / Embedding Extraction

WHAT THIS MODULE DOES:
  Loads the all-MiniLM-L6-v2 transformer model locally and exposes a
  clean function that converts raw text into a 384-dimensional embedding
  vector via PyTorch's forward pass + mean pooling.

  This module is the bridge between raw language and mathematics.
  Once text becomes a vector, every downstream operation — clustering,
  classification, routing — operates purely on numbers.

HOW IT FITS IN THE PIPELINE:
  Raw Text → [THIS MODULE] → 384-dim Vector → K-Means Classifier → Agent Router

ARCHITECTURE DECISION — Why raw transformers instead of sentence-transformers?
  The sentence-transformers library does all of this in one line:
    model.encode("some text")
  But using the raw AutoTokenizer + AutoModel API forces us to understand:
    - What tokenization actually produces (token IDs, attention masks)
    - What the model's forward pass returns (hidden states per token)
    - Why mean pooling exists and how the math works
    - How to handle padding correctly
  That understanding matters when debugging embedding quality in production.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


# =============================================================================
# CONFIGURATION & DEVICE DETECTION
# =============================================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_device() -> torch.device:
    """
    Auto-detect the best available compute device.

    Priority order:
      1. MPS  → Apple Silicon GPU (M1/M2/M3 Macs). Fastest option on Mac.
      2. CUDA → NVIDIA GPU. Fastest on Linux/Windows workstations.
      3. CPU  → Universal fallback. Works everywhere, slightly slower.

    For this project, CPU is more than fast enough — MiniLM-L6-v2 is tiny.
    MPS/CUDA are a bonus if available.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


DEVICE = _get_device()


# =============================================================================
# MODEL CACHE (Singleton Pattern)
# =============================================================================
# These are module-level variables, initialized to None.
# They get populated exactly ONCE on the first call to _load_model(),
# then reused for every subsequent call.
#
# WHY THE SINGLETON PATTERN HERE?
#   The model weights are ~90MB and take ~2-3 seconds to load off disk.
#   If we called AutoModel.from_pretrained() inside get_embedding() directly,
#   we'd pay that 2-3 second cost on every single API request.
#   Module-level caching means we pay it once at startup, then every
#   subsequent embedding is computed in milliseconds.
_tokenizer = None
_model = None


def _load_model() -> None:
    """
    Lazily load the tokenizer and model into module-level cache.

    "Lazily" means we don't load on import — we load on first use.
    This prevents slow startup if the module is imported but never called.
    After the first call, the global check short-circuits instantly.
    """
    global _tokenizer, _model

    # Short-circuit: if already loaded, do nothing
    if _tokenizer is not None and _model is not None:
        return

    print(f"[Embedder] Loading model  : {MODEL_NAME}")
    print(f"[Embedder] Target device  : {DEVICE}")
    print(f"[Embedder] First run downloads ~90MB → ~/.cache/huggingface/")

    # AutoTokenizer loads the vocabulary and tokenization rules.
    # For MiniLM-L6-v2, this is a WordPiece tokenizer with a 30,522-word vocab.
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # AutoModel loads the transformer architecture + pre-trained weights.
    # MiniLM-L6-v2: 6 attention layers, 384 hidden dims, ~22M parameters.
    _model = AutoModel.from_pretrained(MODEL_NAME)

    # Move the model weights to the target device (MPS/CUDA/CPU).
    # All tensor operations must happen on the same device as the model.
    _model = _model.to(DEVICE)

    # Switch to inference mode. This disables two training-only behaviors:
    #   1. Dropout layers (randomly zero out neurons during training for regularization)
    #      → in eval(), dropout is disabled, giving deterministic outputs
    #   2. BatchNorm statistics updates
    #      → in eval(), BatchNorm uses frozen running statistics
    # Without .eval(), you'd get slightly different embeddings on each call
    # for the same input — a subtle but serious bug.
    _model.eval()

    print(f"[Embedder] ✅ Model ready.\n")


# =============================================================================
# MEAN POOLING
# =============================================================================

def _mean_pool(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Collapse a sequence of token vectors into a single sentence vector.

    ── THE PROBLEM ───────────────────────────────────────────────────────────
    After the forward pass, the model outputs one 384-dim vector PER TOKEN.
    For "Where is my order?" (7 tokens), the output shape is (1, 7, 384).
    The K-Means classifier needs exactly ONE vector per sentence.
    We have to reduce (1, 7, 384) → (1, 384).

    ── WHY NOT JUST TAKE THE [CLS] TOKEN? ────────────────────────────────────
    [CLS] is the first token (index 0), and BERT-style models were originally
    trained to use it as a sentence-level summary for classification tasks.
    However, all-MiniLM-L6-v2 was fine-tuned for semantic similarity using
    mean pooling as the aggregation strategy. Using [CLS] here would
    underperform relative to how the model was actually trained.

    ── THE MATH ──────────────────────────────────────────────────────────────
    We want the average of the REAL token vectors, ignoring padding tokens.

    Padding tokens are added to make all sequences in a batch the same length.
    Their attention_mask value is 0 (vs 1 for real tokens). We must exclude
    them from the average, or padding would dilute the meaning.

      token_embeddings : Tensor (batch, seq_len, 384)
      attention_mask   : Tensor (batch, seq_len)     — 1=real, 0=padding

    Step 1 — Expand the mask to match embedding dimensions:
      (batch, seq_len) → (batch, seq_len, 384)
      Each mask value gets copied 384 times along a new last dimension.

    Step 2 — Zero out padding positions:
      token_embeddings * expanded_mask
      Real token rows are unchanged (×1). Padding rows become all zeros (×0).

    Step 3 — Sum across the sequence dimension (dim=1):
      (batch, seq_len, 384) → (batch, 384)

    Step 4 — Divide by the count of real tokens per sample:
      This gives us the true mean, not a padded-diluted mean.
      clamp(min=1e-9) prevents division-by-zero for edge-case empty inputs.

    RETURNS:
      Tensor of shape (batch_size, 384)
    """
    # Step 1: Expand mask from (batch, seq_len) → (batch, seq_len, 384)
    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)                    # (batch, seq_len, 1)
        .expand(token_embeddings.size())  # (batch, seq_len, 384) — broadcast copy
        .float()                          # convert bool/int mask to float for math
    )

    # Step 2 + 3: Zero out padding, then sum across sequence dimension
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)  # (batch, 384)

    # Step 4: Divide by real token count (safe division)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)  # (batch, 384)

    return sum_embeddings / sum_mask  # (batch, 384)


# =============================================================================
# PUBLIC API — SINGLE TEXT
# =============================================================================

def get_embedding(text: str) -> np.ndarray:
    """
    Convert a single raw text string into a 384-dimensional embedding vector.

    This is the function the rest of the system calls at runtime.
    The LangGraph router calls this on every incoming API request.

    FULL PIPELINE:
      text → tokenize → forward pass → mean pool → L2 normalize → numpy

    ARGS:
      text : str  Raw user query (e.g., "I want to cancel my order #1234")

    RETURNS:
      np.ndarray  Shape (384,), dtype float32, L2-normalized (unit length)

    EXAMPLE:
      >>> from app.pipeline.embedder import get_embedding
      >>> vec = get_embedding("cancel my order")
      >>> vec.shape
      (384,)
      >>> import numpy as np
      >>> round(float(np.linalg.norm(vec)), 4)
      1.0
    """
    _load_model()

    # ── STEP 1: TOKENIZATION ──────────────────────────────────────────────────
    # The tokenizer converts raw text into integer IDs from the model's vocabulary.
    #
    # What's actually happening:
    #   "cancel my order" → WordPiece splits → adds [CLS] at start, [SEP] at end
    #   → looks up each piece in the 30k-word vocabulary
    #   → returns integer IDs
    #
    # "cancel my order" might tokenize to something like:
    #   input_ids:      [101, 17542, 2026, 2344, 102]
    #                    CLS  cancel  my   order  SEP
    #   attention_mask: [1,   1,      1,   1,     1]
    #
    # padding=True    : pads shorter sequences in a batch to uniform length
    # truncation=True : cuts sequences longer than max_length (rare for support queries)
    # max_length=128  : MiniLM was trained up to 256 tokens, but 128 covers
    #                   virtually all customer support queries and halves compute cost
    # return_tensors="pt" : return PyTorch tensors, not Python lists
    encoded = _tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Move tensors to the same device as the model
    input_ids      = encoded["input_ids"].to(DEVICE)       # (1, seq_len)
    attention_mask = encoded["attention_mask"].to(DEVICE)  # (1, seq_len)

    # ── STEP 2: FORWARD PASS ─────────────────────────────────────────────────
    # Run the token IDs through all 6 transformer attention layers.
    #
    # torch.no_grad() context manager:
    #   During training, PyTorch builds a computation graph to enable
    #   backpropagation (gradient calculation). We don't need gradients
    #   for inference — disabling them cuts memory usage in half and
    #   speeds up the forward pass.
    with torch.no_grad():
        outputs = _model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # outputs.last_hidden_state: (1, seq_len, 384)
    # This is the contextual representation of EVERY token after attending
    # to all other tokens in the sequence. The word "cancel" in
    # "cancel my order" has a different vector than "cancel" in
    # "cancel culture" — context is baked in.
    token_embeddings = outputs.last_hidden_state  # (1, seq_len, 384)

    # ── STEP 3: MEAN POOLING ─────────────────────────────────────────────────
    # Collapse (1, seq_len, 384) → (1, 384)
    # See _mean_pool() docstring above for the full explanation.
    pooled = _mean_pool(token_embeddings, attention_mask)  # (1, 384)

    # ── STEP 4: L2 NORMALIZATION ─────────────────────────────────────────────
    # Scale the vector so its Euclidean length equals exactly 1.0.
    # This places the vector on the surface of a 384-dimensional unit sphere.
    #
    # WHY NORMALIZE?
    #   K-Means minimizes Euclidean distance between points and centroids.
    #   For unnormalized vectors, a longer sentence would produce a larger
    #   magnitude vector — not because it means something more extreme,
    #   but just because more tokens were summed. This would corrupt the
    #   distance calculations.
    #
    #   After normalization, distance is purely about direction (meaning),
    #   not magnitude (sentence length). Two paraphrases will be close
    #   together regardless of word count.
    #
    # p=2, dim=1 : L2 norm (Euclidean), applied along the feature dimension
    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)  # (1, 384)

    # ── STEP 5: TENSOR → NUMPY ───────────────────────────────────────────────
    # Scikit-learn (Phase 3) operates on NumPy arrays, not PyTorch tensors.
    #
    # .squeeze(0) : removes the batch dimension (1, 384) → (384,)
    # .cpu()      : moves tensor from MPS/CUDA back to CPU RAM
    #               (NumPy only knows about CPU memory)
    # .numpy()    : zero-copy conversion from PyTorch tensor to NumPy array
    embedding = normalized.squeeze(0).cpu().numpy()  # (384,) float32

    return embedding


# =============================================================================
# PUBLIC API — BATCH (used by extract_embeddings.py for training data)
# =============================================================================

def get_embeddings_batch(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Efficiently embed a large list of texts by processing in mini-batches.

    WHY NOT JUST LOOP OVER get_embedding()?
      Each call to get_embedding() processes 1 text in 1 forward pass.
      Batching groups N texts into a SINGLE forward pass.
      Modern hardware (GPU/MPS/even CPU) can parallelize across the batch
      dimension — it's doing the same math N times simultaneously rather
      than N times sequentially. For 3,000 texts, this is 10-20x faster.

    MEMORY NOTE:
      We convert each batch to NumPy immediately after the forward pass
      and release the GPU tensors. This keeps GPU memory usage flat
      regardless of dataset size.

    ARGS:
      texts      : list[str]  — list of raw text strings to embed
      batch_size : int        — texts per forward pass (64 is safe for CPU/MPS)

    RETURNS:
      np.ndarray of shape (len(texts), 384), dtype float32
    """
    _load_model()

    all_embeddings = []
    n = len(texts)
    total_batches = (n + batch_size - 1) // batch_size  # ceiling division

    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        current_batch = i // batch_size + 1

        print(
            f"\r[Embedder] Batch {current_batch}/{total_batches} "
            f"| {min(i + batch_size, n)}/{n} texts",
            end="",
            flush=True
        )

        # Same pipeline as get_embedding(), but operating on a list of strings.
        # The tokenizer automatically pads all texts in the batch to the same
        # length (the longest sequence in the batch), so each batch may have
        # a different seq_len — that's fine and expected.
        encoded = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids      = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = _model(input_ids=input_ids, attention_mask=attention_mask)

        token_embeddings = outputs.last_hidden_state
        pooled           = _mean_pool(token_embeddings, attention_mask)
        normalized       = torch.nn.functional.normalize(pooled, p=2, dim=1)

        # Convert to NumPy and release GPU memory for this batch
        batch_embeddings = normalized.cpu().numpy()  # (batch_size, 384)
        all_embeddings.append(batch_embeddings)

    print()  # move to new line after progress output

    # np.vstack: stack all (batch_size, 384) arrays vertically
    # Result shape: (total_texts, 384)
    return np.vstack(all_embeddings)