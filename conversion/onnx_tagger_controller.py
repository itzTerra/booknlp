import torch
import numpy as np
from pathlib import Path
from typing import Dict
import sys
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent.parent))


class ONNXTaggerController:
    """
    Unified controller for ONNX-based tagger inference.

    All torch layers with weights are exported to ONNX:
    - BERT encoder + embedding reduction
    - Entity head: 3-layer LSTM + CRF with Viterbi decoding
    - Supersense head: 1-layer LSTM with WordNet embeddings
    - Event head: Flat LSTM + classifier

    The ONNX model returns final predictions (not logits).
    No post-processing needed - predictions are ready to use.
    """

    def __init__(
        self,
        onnx_model_path: str,
        device: str = "cpu",
    ):
        """
        Initialize controller with ONNX model.

        Args:
            onnx_model_path: Path to full ONNX model with all layers
            device: Device for PyTorch computations ('cpu' or 'cuda')
        """
        self.device = device

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers.insert(0, "CUDAExecutionProvider")

        self.onnx_session = ort.InferenceSession(onnx_model_path, providers=providers)

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        transforms: torch.Tensor,
        matrix1: torch.Tensor,
        matrix2: torch.Tensor,
        wn: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run complete prediction pipeline.

        All inference happens in ONNX. Outputs are final predictions.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            transforms: Transformation matrix
            matrix1: Entity layer 1-to-2 transformation
            matrix2: Entity layer 2-to-3 transformation
            wn: WordNet sense IDs
            seq_lengths: Actual sequence lengths

        Returns:
            Dictionary with predictions for enabled heads
        """
        ort_inputs = {
            "input_ids": input_ids.cpu().numpy().astype(np.int64),
            "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
            "transforms": transforms.cpu().numpy().astype(np.float32),
            "matrix1": matrix1.cpu().numpy().astype(np.float32),
            "matrix2": matrix2.cpu().numpy().astype(np.float32),
            "wn": wn.cpu().numpy().astype(np.int64),
            "seq_lengths": seq_lengths.cpu().numpy().astype(np.int64),
        }

        ort_outputs = self.onnx_session.run(None, ort_inputs)

        entity_preds1 = torch.from_numpy(ort_outputs[0]).to(self.device)
        entity_preds2 = torch.from_numpy(ort_outputs[1]).to(self.device)
        entity_preds3 = torch.from_numpy(ort_outputs[2]).to(self.device)
        supersense_preds = torch.from_numpy(ort_outputs[3]).to(self.device)
        event_preds = torch.from_numpy(ort_outputs[4]).to(self.device)

        results = {
            "entity": (entity_preds1, entity_preds2, entity_preds3),
            "supersense": supersense_preds,
            "event": event_preds,
        }

        return results


def create_controller_from_onnx_model(
    onnx_model_path: str,
    device: str = "cpu",
) -> ONNXTaggerController:
    """
    Create controller from ONNX model path.

    Args:
        onnx_model_path: Path to full ONNX model with all weighted layers
        device: Device for PyTorch computations

    Returns:
        Initialized ONNXTaggerController
    """
    controller = ONNXTaggerController(onnx_model_path, device)
    return controller
