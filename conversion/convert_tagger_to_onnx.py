"""
Convert Tagger model to ONNX format.

Exports all torch layers with weights to a single unified ONNX model.
This includes BERT, all three task heads, and embedding layers.

Architecture:
- ONNX Model: Complete tagger with all weighted layers (logits only)
- External Decoder: CRF Viterbi decoding using exported transitions
"""

from pathlib import Path
import json
import sys
from typing import Dict, List, Optional, Tuple
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
import torch
import torch.nn as nn

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from booknlp.english.tagger import Tagger


class TaggerLogitsWrapper(nn.Module):
    """
    Tagger wrapper that exports logits only (no CRF decoding).

    Outputs logits for all heads so decoding can be handled externally.

    Note: Transform matrices are asymmetric [batch, original_tokens, wordpiece_tokens]
    since they reduce BERT wordpiece outputs back to original token space.
    """

    def __init__(self, tagger: Tagger):
        super(TaggerLogitsWrapper, self).__init__()
        self.bert = tagger.bert
        self.num_layers = tagger.num_layers
        self.layered_dropout = tagger.layered_dropout

        # Entity head layers
        self.lstm1 = tagger.lstm1
        self.lstm2 = tagger.lstm2
        self.lstm3 = tagger.lstm3
        self.hidden2tag1 = tagger.hidden2tag1
        self.hidden2tag2 = tagger.hidden2tag2
        self.hidden2tag3 = tagger.hidden2tag3

        # Supersense head layers
        self.supersense_lstm1 = tagger.supersense_lstm1
        self.supersense_hidden2tag1 = tagger.supersense_hidden2tag1
        self.wn_embedding = tagger.wn_embedding

        # Event head layers
        self.flat_lstm = tagger.flat_lstm
        self.flat_classifier = tagger.flat_classifier

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        transforms: torch.Tensor,
        matrix1: torch.Tensor,
        matrix2: torch.Tensor,
        wn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all heads, returning logits only.

        Args:
            input_ids: Token IDs from BERT tokenizer
            attention_mask: Attention mask for BERT
            transforms: Transformation matrix from token embeddings
            matrix1: Entity layer 1-to-2 transformation matrix
            matrix2: Entity layer 2-to-3 transformation matrix
            wn: WordNet sense IDs

        Returns:
            Tuple of (entity_logits1, entity_logits2, entity_logits3, supersense_logits, event_logits)
        """
        # Get BERT embeddings and reduce
        output = self.bert(
            input_ids,
            token_type_ids=None,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = output["hidden_states"]

        if self.num_layers == 4:
            all_layers = torch.cat(
                (
                    hidden_states[-1],
                    hidden_states[-2],
                    hidden_states[-3],
                    hidden_states[-4],
                ),
                2,
            )
        elif self.num_layers == 2:
            all_layers = torch.cat((hidden_states[-1], hidden_states[-2]), 2)

        reduced = torch.matmul(transforms, all_layers)[:, 1:, :]

        # Entity head - output tag logits at each layer
        reduced = self.layered_dropout(reduced)
        lstm_out1, _ = self.lstm1(reduced)
        tag_space1 = self.hidden2tag1(lstm_out1)

        input2 = torch.matmul(matrix1[:, 1:, 1:], lstm_out1)
        input2 = self.layered_dropout(input2)
        lstm_out2, _ = self.lstm2(input2)
        tag_space2 = self.hidden2tag2(lstm_out2)

        input3 = torch.matmul(matrix2[:, 1:, 1:], lstm_out2)
        input3 = self.layered_dropout(input3)
        lstm_out3, _ = self.lstm3(input3)
        tag_space3 = self.hidden2tag3(lstm_out3)

        # Supersense head - output logits
        wn_embeds = self.wn_embedding(wn)
        wn_embeds = wn_embeds[:, 1:, :]
        combined = torch.cat([reduced, wn_embeds], axis=2)
        combined = self.layered_dropout(combined)
        lstm_out_ss, _ = self.supersense_lstm1(combined)
        tag_space_ss = self.supersense_hidden2tag1(lstm_out_ss)

        # Event head - output logits
        out, _ = self.flat_lstm(reduced)
        out_flat = out.contiguous().view(-1, out.shape[2])
        event_logits = self.flat_classifier(out_flat)
        event_logits = event_logits.view(out.shape[0], out.shape[1], -1)

        return (
            tag_space1,
            tag_space2,
            tag_space3,
            tag_space_ss,
            event_logits,
        )


def export_crf_transitions(tagger: Tagger, output_dir: Path) -> Path:
    """
    Export CRF transition matrices for external Viterbi decoding.

    Args:
        tagger: Loaded Tagger instance containing CRF parameters.
        output_dir: Directory to write the JSON artifact to.

    Returns:
        Path to the JSON file containing transition matrices.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    entity_transitions = tagger.crf.transitions.detach().cpu().numpy()
    supersense_transitions = tagger.supersense_crf.transitions.detach().cpu().numpy()

    payload = {
        "entity_transitions": entity_transitions.tolist(),
        "supersense_transitions": supersense_transitions.tolist(),
        "entity_num_labels": int(entity_transitions.shape[0]),
        "supersense_num_labels": int(supersense_transitions.shape[0]),
        "entity_start_idx": int(entity_transitions.shape[0] - 2),
        "entity_stop_idx": int(entity_transitions.shape[0] - 1),
        "supersense_start_idx": int(supersense_transitions.shape[0] - 2),
        "supersense_stop_idx": int(supersense_transitions.shape[0] - 1),
    }

    output_path = output_dir / "crf_transitions.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    return output_path


def convert_tagger_to_onnx(
    tagger: Tagger,
    output_dir: Path = None,
    opset_version: int = 18,
) -> Path:
    """
    Convert full Tagger model to ONNX format.

    Exports all torch layers with weights (logits only, no decoding):
    - BERT encoder + embedding reduction
    - Entity head: 3-layer LSTM + logits
    - Supersense head: 1-layer LSTM + logits
    - Event head: Flat LSTM + logits

    Args:
        tagger: Trained Tagger model
        output_dir: Directory to save ONNX model
        opset_version: ONNX opset version (14+ recommended)

    Returns:
        Path to the exported ONNX model
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "models"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Converting full tagger model to ONNX format...")
    print(f"Output directory: {output_dir}")

    device = tagger.device if hasattr(tagger, "device") else "cpu"
    batch_size = 1
    wordpiece_seq_len = 512
    original_seq_len = 256

    wrapper = TaggerLogitsWrapper(tagger).to(device)
    wrapper.eval()

    dummy_input_ids = torch.randint(
        0, 1000, (batch_size, wordpiece_seq_len), device=device, dtype=torch.long
    )
    dummy_attention_mask = torch.ones(
        (batch_size, wordpiece_seq_len), device=device, dtype=torch.long
    )
    dummy_transforms = torch.randn(
        batch_size, original_seq_len, wordpiece_seq_len, device=device
    )
    dummy_matrix1 = torch.randn(
        batch_size, original_seq_len, original_seq_len, device=device
    )
    dummy_matrix2 = torch.randn(
        batch_size, original_seq_len, original_seq_len, device=device
    )
    dummy_wn = torch.randint(
        0, 50, (batch_size, original_seq_len), device=device, dtype=torch.long
    )

    model_path = output_dir / "model.onnx"

    torch.onnx.export(
        wrapper,
        (
            dummy_input_ids,
            dummy_attention_mask,
            dummy_transforms,
            dummy_matrix1,
            dummy_matrix2,
            dummy_wn,
        ),
        str(model_path),
        input_names=[
            "input_ids",
            "attention_mask",
            "transforms",
            "matrix1",
            "matrix2",
            "wn",
        ],
        output_names=[
            "entity_logits1",
            "entity_logits2",
            "entity_logits3",
            "supersense_logits",
            "event_logits",
        ],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "wordpiece_seq"},
            "attention_mask": {0: "batch", 1: "wordpiece_seq"},
            "transforms": {0: "batch", 1: "original_seq", 2: "wordpiece_seq"},
            "matrix1": {0: "batch", 1: "original_seq", 2: "original_seq"},
            "matrix2": {0: "batch", 1: "original_seq", 2: "original_seq"},
            "wn": {0: "batch", 1: "original_seq"},
            "entity_logits1": {0: "batch", 1: "original_seq"},
            "entity_logits2": {0: "batch", 1: "original_seq"},
            "entity_logits3": {0: "batch", 1: "original_seq"},
            "supersense_logits": {0: "batch", 1: "original_seq"},
            "event_logits": {0: "batch", 1: "original_seq"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
        dynamo=False,
    )

    print(f"✓ Full tagger model exported to {model_path}")
    print("\nModel inputs (with corrected asymmetric transforms):")
    print("  - input_ids: [batch, wordpiece_seq] - BERT tokenized input")
    print("  - attention_mask: [batch, wordpiece_seq] - BERT attention mask")
    print(
        "  - transforms: [batch, original_seq, wordpiece_seq] - reduces wordpiece to original tokens"
    )
    print(
        "  - matrix1: [batch, original_seq, original_seq] - layer1-to-layer2 transformation"
    )
    print(
        "  - matrix2: [batch, original_seq, original_seq] - layer2-to-layer3 transformation"
    )
    print("  - wn: [batch, original_seq] - WordNet sense IDs")
    print("\nModel outputs (logits only):")
    print(
        "  - entity_logits1, entity_logits2, entity_logits3: [batch, original_seq, labels]"
    )
    print("  - supersense_logits: [batch, original_seq, labels]")
    print("  - event_logits: [batch, original_seq, 2]")

    return model_path


def resolve_checkpoint_path(candidate: Optional[str]) -> Path:
    """
    Resolve the checkpoint path from an explicit argument or the conversion dir.

    Args:
        candidate: Optional checkpoint path provided by the user.

    Returns:
        Path to the resolved checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint can be resolved.
        ValueError: If multiple checkpoints are found without an explicit path.
    """
    if candidate:
        path = Path(candidate)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")

    conversion_dir = Path(__file__).parent
    matches = sorted(conversion_dir.glob("*.model"))
    if not matches:
        raise FileNotFoundError(
            "No checkpoint found. Provide a .model path or place one in conversion/."
        )
    if len(matches) > 1:
        raise ValueError(
            "Multiple checkpoints found in conversion/. Provide the exact path."
        )
    return matches[0]


def export_transformers_artifacts(tagger: Tagger, output_dir: Path) -> Dict[str, Path]:
    """
    Export Transformers config and tokenizer artifacts alongside the ONNX model.

    This mirrors the non-weight files typically present in Hugging Face repos and
    avoids exporting PyTorch weights since ONNX already contains the parameters.

    Args:
        tagger: Loaded Tagger instance with a configured BERT model and tokenizer.
        output_dir: Directory to write exported artifacts into.

    Returns:
        Mapping of artifact file names to their paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files: Dict[str, Path] = {}

    config_files = tagger.bert.config.save_pretrained(str(output_dir))
    if config_files:
        for file_path in config_files:
            path = Path(file_path)
            saved_files[path.name] = path
    else:
        config_path = output_dir / "config.json"
        if config_path.exists():
            saved_files[config_path.name] = config_path

    tokenizer_files = tagger.tokenizer.save_pretrained(str(output_dir))
    if tokenizer_files:
        for file_path in tokenizer_files:
            path = Path(file_path)
            saved_files[path.name] = path

    vocab_files = tagger.tokenizer.save_vocabulary(str(output_dir))
    if vocab_files:
        for file_path in vocab_files:
            path = Path(file_path)
            saved_files[path.name] = path

    return saved_files


def export_fp16_onnx_with_optimum(onnx_model_path: Path) -> Optional[Path]:
    """
    Export an FP16-optimized ONNX model using Optimum.

    This step keeps the base ONNX export intact and creates an additional
    FP16-optimized model with the _fp16 suffix for faster inference.

    Args:
        onnx_model_path: Path to the base ONNX model to optimize.

    Returns:
        Path to the FP16 ONNX model if created, otherwise None.
    """
    if not onnx_model_path.exists():
        print(f"Warning: Base ONNX model not found at {onnx_model_path}")
        return None

    save_dir = onnx_model_path.parent

    try:
        optimizer = ORTOptimizer.from_pretrained(onnx_model_path.parent)
    except Exception as exc:
        print(f"Warning: Optimum optimizer load failed: {exc}")
        return None

    optimization_config = OptimizationConfig(optimization_level=1, fp16=True)
    try:
        optimizer.optimize(
            save_dir=save_dir,
            file_suffix="fp16",
            optimization_config=optimization_config,
        )
    except Exception as exc:
        print(f"Warning: FP16 optimization failed: {exc}")
        return None

    model_path = save_dir / "model_fp16.onnx"
    if not model_path.exists():
        print("Warning: FP16 model export did not produce model_fp16.onnx")
        return None

    print(f"✓ FP16 ONNX model exported to {model_path}")
    return model_path


def export_q8_onnx(onnx_model_path: Path) -> Optional[Path]:
    """
    Export an INT8 (q8) quantized ONNX model using ONNX Runtime quantization.

    This performs dynamic quantization of weights to QInt8 which is commonly
    called "q8" in many toolchains. The output file is written alongside the
    base model as `model_quantized.onnx`.

    Args:
        onnx_model_path: Path to the base ONNX model to quantize.

    Returns:
        Path to the quantized model if created, otherwise None.
    """
    if not onnx_model_path.exists():
        print(f"Warning: Base ONNX model not found at {onnx_model_path}")
        return None

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as exc:
        print(f"Warning: onnxruntime.quantization not available: {exc}")
        return None

    save_dir = onnx_model_path.parent
    out_path = save_dir / "model_quantized.onnx"

    try:
        # Quantize weights to QInt8 (8-bit quantization)
        quantize_dynamic(
            model_input=str(onnx_model_path),
            model_output=str(out_path),
            weight_type=QuantType.QInt8,
        )
    except Exception as exc:
        print(f"Warning: Q8 quantization failed: {exc}")
        return None

    if not out_path.exists():
        print("Warning: Q8 quantization did not produce model_quantized.onnx")
        return None

    print(f"✓ Q8 ONNX model exported to {out_path}")
    return out_path


def validate_onnx_model(
    tagger: Tagger,
    onnx_model_path: Path,
    tolerance: float = 1e-4,
    model_name: str = "ONNX",
    test_shapes: Optional[list] = None,
) -> bool:
    """
    Validate ONNX model produces same outputs as PyTorch model.

    Tests with multiple batch sizes and sequence lengths to ensure dynamic axes work.

    Args:
        tagger: Original PyTorch tagger
        onnx_model_path: Path to exported ONNX model
        tolerance: Maximum allowed difference (not used for discrete outputs)
        model_name: Name of the model for logging purposes
        test_shapes: List of (batch_size, seq_len) tuples to test. If None, tests multiple shapes.

    Returns:
        True if validation passes
    """
    try:
        import onnxruntime as ort
        import onnx
    except ImportError:
        print("Warning: onnxruntime not installed, skipping validation")
        return False

    # Check ONNX model validity
    try:
        model = onnx.load(str(onnx_model_path))
        onnx.checker.check_model(model)
        print(f"✓ {model_name} model passed ONNX checker")
    except Exception as e:
        print(f"✗ {model_name} model failed ONNX checker: {e}")
        return False

    print(
        f"\nValidating {model_name} model with dynamic batch sizes and sequence lengths..."
    )

    device = tagger.device if hasattr(tagger, "device") else "cpu"
    ort_session = ort.InferenceSession(
        str(onnx_model_path), providers=["CPUExecutionProvider"]
    )

    wrapper = TaggerLogitsWrapper(tagger).to(device)
    wrapper.eval()

    # Test with multiple shapes to ensure dynamic batch size and sequence length work
    if test_shapes is None:
        test_shapes = [
            (1, 128),  # Single sample, medium length
            (2, 128),  # Batch of 2
            (4, 100),  # Batch of 4
            (32, 52),  # Real-world batch size
        ]
    all_passed = True
    for batch_size, actual_seq_len in test_shapes:
        print(f"\n  Testing shape: batch_size={batch_size}, seq_len={actual_seq_len}")

        dummy_input_ids = torch.randint(
            0, 1000, (batch_size, actual_seq_len), device=device, dtype=torch.long
        )
        dummy_attention_mask = torch.ones(
            (batch_size, actual_seq_len), device=device, dtype=torch.long
        )

        dummy_transforms = torch.randn(
            batch_size, actual_seq_len, actual_seq_len, device=device
        )
        dummy_matrix1 = torch.randn(
            batch_size, actual_seq_len, actual_seq_len, device=device
        )
        dummy_matrix2 = torch.randn(
            batch_size, actual_seq_len, actual_seq_len, device=device
        )
        dummy_wn = torch.randint(
            0, 50, (batch_size, actual_seq_len), device=device, dtype=torch.long
        )

        with torch.no_grad():
            pytorch_outputs = wrapper(
                dummy_input_ids,
                dummy_attention_mask,
                dummy_transforms,
                dummy_matrix1,
                dummy_matrix2,
                dummy_wn,
            )

        ort_inputs = {
            "input_ids": dummy_input_ids.cpu().numpy().astype(np.int64),
            "attention_mask": dummy_attention_mask.cpu().numpy().astype(np.int64),
            "transforms": dummy_transforms.cpu().numpy().astype(np.float32),
            "matrix1": dummy_matrix1.cpu().numpy().astype(np.float32),
            "matrix2": dummy_matrix2.cpu().numpy().astype(np.float32),
            "wn": dummy_wn.cpu().numpy().astype(np.int64),
        }

        try:
            onnx_outputs = ort_session.run(None, ort_inputs)
        except Exception as e:
            print(f"    ✗ FAILED to run inference: {e}")
            all_passed = False
            continue

        output_names = [
            "entity_logits1",
            "entity_logits2",
            "entity_logits3",
            "supersense_logits",
            "event_logits",
        ]

        shape_passed = True
        for pytorch_out, onnx_out, name in zip(
            pytorch_outputs, onnx_outputs, output_names
        ):
            pytorch_np = pytorch_out.cpu().numpy()
            max_diff = np.max(np.abs(pytorch_np - onnx_out))

            if not np.allclose(pytorch_np, onnx_out, rtol=1e-3, atol=1e-3):
                print(f"      {name}: max |Δ|={max_diff:.6f} ✗")
                shape_passed = False
            else:
                print(f"      {name}: max |Δ|={max_diff:.6f} ✓")

        if not shape_passed:
            all_passed = False

    return all_passed


def validate_all_model_variants(
    tagger: Tagger,
    output_dir: Path,
    skip_validation: bool = False,
    test_shapes: Optional[List[Tuple[int, int]]] = None,
) -> bool:
    """
    Validate all available model variants (base, FP16).

    Args:
        tagger: Original PyTorch tagger
        output_dir: Directory containing ONNX models
        skip_validation: If True, skip validation
        test_shapes: List of (batch_size, seq_len) tuples to test with. If None, tests multiple shapes.

    Returns:
        True if all available models pass validation
    """
    if skip_validation:
        return True

    output_dir = Path(output_dir)
    onnx_dir = output_dir / "onnx"

    model_variants = [
        ("model.onnx", "Base ONNX"),
        ("model_fp16.onnx", "FP16 ONNX"),
        ("model_quantized.onnx", "Q8 ONNX"),
    ]

    all_passed = True
    for model_file, model_label in model_variants:
        # Check both root output_dir and onnx subdirectory
        model_path = output_dir / model_file
        if not model_path.exists():
            model_path = onnx_dir / model_file

        if model_path.exists():
            passed = validate_onnx_model(
                tagger, model_path, model_name=model_label, test_shapes=test_shapes
            )
            if not passed:
                all_passed = False
        else:
            print(f"\n⊘ {model_label} model not found at {model_path}")

    return all_passed


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Convert Tagger to ONNX")
    parser.add_argument(
        "checkpoint_path",
        nargs="?",
        help="Path to tagger checkpoint (optional if a single .model is in conversion/)",
    )
    parser.add_argument(
        "--output-dir",
        default="export",
        help="Output directory for ONNX model (default: export/)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use local cached Transformers files (no network)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing ONNX model without re-exporting",
    )
    parser.add_argument(
        "--export-fp16",
        action="store_true",
        help="Export FP16 optimized ONNX model using Optimum",
    )
    parser.add_argument(
        "--export-q8",
        action="store_true",
        help="Export Q8 (INT8) quantized ONNX model using onnxruntime",
    )

    args = parser.parse_args()

    print("ONNX Conversion Script for Tagger Model")
    print("=" * 50)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")

    print("\nLoading tagger checkpoint...")
    tagger = Tagger.load(
        str(checkpoint_path),
        device=args.device,
        local_files_only=args.local_files_only,
    )
    tagger.eval()

    print("✓ Tagger loaded successfully\n")

    if args.validate_only:
        onnx_path = Path(args.output_dir) / "model.onnx"
        if not onnx_path.exists():
            print(f"Error: ONNX model not found at {onnx_path}")
            print("Run without --validate-only to export the model first.")
            sys.exit(1)
        print(f"Validating existing ONNX model at {onnx_path}\n")
    else:
        onnx_path = convert_tagger_to_onnx(tagger, Path(args.output_dir))

    artifacts = export_transformers_artifacts(tagger, Path(args.output_dir))
    if artifacts:
        print("\nExported Transformers artifacts:")
        for name in sorted(artifacts.keys()):
            print(f"  - {name}")

    transition_path = export_crf_transitions(tagger, Path(args.output_dir))
    print(f"\nExported CRF transitions: {transition_path}")

    fp16_onnx_path = None
    if args.export_fp16:
        fp16_onnx_path = export_fp16_onnx_with_optimum(onnx_path)

    q8_onnx_path = None
    if args.export_q8:
        q8_onnx_path = export_q8_onnx(onnx_path)

    if not args.skip_validation:
        validate_all_model_variants(
            tagger, Path(args.output_dir), skip_validation=False
        )

    if args.validate_only:
        print("\n✓ Validation complete!")
    else:
        print("\n✓ Conversion complete!")
    print(f"ONNX model: {onnx_path}")
    if fp16_onnx_path:
        print(f"FP16 ONNX model: {fp16_onnx_path}")
    if q8_onnx_path:
        print(f"Q8 ONNX model: {q8_onnx_path}")
