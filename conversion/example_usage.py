"""
Example usage of ONNX Tagger Controller

This demonstrates how to use the ONNX-accelerated tagger for inference.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conversion.onnx_tagger_controller import create_controller_from_checkpoint
import torch


def main():
    checkpoint_path = "path/to/tagger.pt"
    onnx_model_path = "models/tagger_bert_core.onnx"
    device = "cpu"

    print("Loading controller...")
    controller = create_controller_from_checkpoint(
        checkpoint_path=checkpoint_path,
        onnx_model_path=onnx_model_path,
        device=device,
    )
    print("✓ Controller loaded\n")

    batch_size = 1
    seq_len = 128

    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones((batch_size, seq_len))
    dummy_transforms = torch.randn(batch_size, seq_len, 768 * 4)
    dummy_matrix1 = torch.eye(seq_len).unsqueeze(0)
    dummy_matrix2 = torch.eye(seq_len).unsqueeze(0)
    dummy_wn = torch.zeros((batch_size, seq_len), dtype=torch.long)
    dummy_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)

    print("Running inference...")
    results = controller.predict(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        transforms=dummy_transforms,
        matrix1=dummy_matrix1,
        matrix2=dummy_matrix2,
        wn=dummy_wn,
        mask=dummy_mask,
        run_entity=True,
        run_supersense=True,
        run_event=True,
    )

    print("✓ Inference complete\n")
    print("Results:")
    if "entity" in results:
        preds1, preds2, preds3 = results["entity"]
        print(f"  Entity layer 1 shape: {preds1.shape}")
        print(f"  Entity layer 2 shape: {preds2.shape}")
        print(f"  Entity layer 3 shape: {preds3.shape}")

    if "supersense" in results:
        print(f"  Supersense shape: {results['supersense'].shape}")

    if "event" in results:
        print(f"  Event shape: {results['event'].shape}")


if __name__ == "__main__":
    print("ONNX Tagger Controller - Example Usage")
    print("=" * 50)
    print("\nNote: Update checkpoint_path and onnx_model_path before running\n")

    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease update the paths in this script:")
        print("  - checkpoint_path: Path to your trained tagger checkpoint")
        print("  - onnx_model_path: Path to exported ONNX model")
        print("\nTo create the ONNX model, run:")
        print("  python convert_tagger_to_onnx.py <checkpoint_path>")
