#!/usr/bin/env python3
"""Measure entity tagging time over texts in examples/speed.json.

Usage: python examples/time_tagger_batch.py [path/to/speed.json]
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from booknlp.booknlp import BookNLP

# Optional ONNX timing support
try:
    import numpy as np
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort

    _ONNX_AVAILABLE = True
except Exception:
    _ONNX_AVAILABLE = False


def load_texts(json_path):
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    texts = [item.get("text", "") for item in data]
    return texts


def call_process_sequential(booknlp, texts):
    t0 = time.perf_counter()
    results = []
    for i, text in enumerate(texts):
        try:
            res = booknlp.process(text=text, out_folder=None, doc_id=f"text_{i}")
        except Exception as e:
            res = None
            print(f"Error processing text {i}: {e}")
        t1 = time.perf_counter()
        results.append(res)
    elapsed = t1 - t0
    return results, elapsed


def call_onnx_sequential(model_repo: str, texts, seq_len: int = 64):
    """Load ONNX model (from HuggingFace repo or local path) and run sequential inferences.

    This constructs minimal dummy inputs matching the expected ONNX input names
    used by the TypeScript controller (`input_ids`, `attention_mask`, `transforms`,
    `matrix1`, `matrix2`, `wn`) and runs the session once per text to measure
    sequential latency.
    """
    if not _ONNX_AVAILABLE:
        print(
            "ONNX timing skipped: missing dependencies (onnxruntime, huggingface_hub, numpy)"
        )
        return None, None

    # Try to download model.onnx from the repo's 'onnx' subfolder, otherwise treat as local path
    try:
        model_file = hf_hub_download(repo_id=model_repo, filename="onnx/model.onnx")
    except Exception:
        # assume model_repo is a local directory or path to model.onnx
        candidate = Path(model_repo)
        if candidate.is_dir():
            model_file = str(candidate / "onnx" / "model.onnx")
        else:
            model_file = str(candidate)

    if not Path(model_file).exists():
        print(f"ONNX model not found at {model_file}; skipping ONNX timing")
        return None, None

    try:
        sess = ort.InferenceSession(model_file)
    except Exception as e:
        print(f"Failed to create ONNX InferenceSession: {e}")
        return None, None

    # Prepare fixed-size dummy inputs (batch_size=1)
    batch_size = 1
    wordpiece_seq_len = seq_len
    original_seq_len = seq_len

    input_ids = np.arange(1, wordpiece_seq_len + 1, dtype=np.int64).reshape(
        batch_size, wordpiece_seq_len
    )
    attention_mask = np.ones((batch_size, wordpiece_seq_len), dtype=np.int64)
    transforms = np.zeros(
        (batch_size, original_seq_len, wordpiece_seq_len), dtype=np.float32
    )
    matrix1 = np.zeros(
        (batch_size, original_seq_len, original_seq_len), dtype=np.float32
    )
    matrix2 = np.zeros(
        (batch_size, original_seq_len, original_seq_len), dtype=np.float32
    )
    wn = np.zeros((batch_size, original_seq_len), dtype=np.int64)

    feeds = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "transforms": transforms,
        "matrix1": matrix1,
        "matrix2": matrix2,
        "wn": wn,
    }

    t0 = time.perf_counter()
    last_t = t0
    results = []
    for i, _text in enumerate(texts):
        try:
            out = sess.run(None, feeds)
            results.append(out)
        except Exception as e:
            print(f"ONNX inference error for text {i}: {e}")
            results.append(None)
        last_t = time.perf_counter()

    elapsed = last_t - t0
    return results, elapsed


def main():
    json_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).with_name("speed.json")
    )
    if not json_path.exists():
        # try examples/ directory
        json_path = Path(__file__).parent / "speed.json"
    if not json_path.exists():
        print("speed.json not found. Provide path as first arg.")
        sys.exit(1)

    texts = load_texts(json_path)
    print(f"Loaded {len(texts)} texts from {json_path}")

    model_params = {"pipeline": "entity", "model": "small", "verbose": False}
    booknlp = BookNLP("en", model_params)

    print("Processing texts sequentially with BookNLP.process...")
    results, elapsed = call_process_sequential(booknlp, texts)

    print(f"Processed {len(texts)} texts in {elapsed:.3f}s")
    if len(texts) > 0:
        print(f"Average per-text: {elapsed / len(texts):.3f}s")


if __name__ == "__main__":
    main()
