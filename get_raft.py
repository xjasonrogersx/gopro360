import sys
import os
import argparse
from pathlib import Path

# 1. Add both RAFT root and RAFT/core to sys.path so internal imports resolve
raft_root = Path(__file__).resolve().parent / "RAFT"
raft_core = raft_root / "core"

if not raft_core.exists():
    raft_root = Path.cwd() / "RAFT"
    raft_core = raft_root / "core"

if not raft_core.exists():
    raise ModuleNotFoundError(
        f"Could not find RAFT core directory at '{raft_core}'. "
        "Ensure the official RAFT repo is cloned into your project folder."
    )

sys.path.insert(0, str(raft_root))
sys.path.insert(0, str(raft_core))

import torch
import numpy as np
import onnx
import onnxruntime as ort
from core.raft import RAFT


def export_and_validate_raft_onnx():
    # Use argparse.Namespace so membership checks like `'dropout' in args` work
    args = argparse.Namespace(
        small=True,
        mixed_precision=False,
        alternate_corr=False,
        dropout=0.0
    )

    model = RAFT(args)
    model.eval()

    img1 = torch.randn(1, 3, 360, 480)
    img2 = torch.randn(1, 3, 360, 480)

    class RAFTExportWrapper(torch.nn.Module):
        def __init__(self, raft_model, iters=12):
            super().__init__()
            self.raft = raft_model
            self.iters = iters

        def forward(self, image1, image2):
            flow_predictions = self.raft(
                image1, image2, iters=self.iters, test_mode=True
            )
            if isinstance(flow_predictions, (tuple, list)):
                return flow_predictions[-1]
            return flow_predictions

    wrapped_model = RAFTExportWrapper(model, iters=12)
    output_path = "raft_light_small_360x480.onnx"

    print("Exporting RAFT-Small model to ONNX...")
    torch.onnx.export(
        wrapped_model,
        (img1, img2),
        output_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["image1", "image2"],
        output_names=["optical_flow"],
        dynamic_axes={
            "image1": {0: "batch_size", 2: "height", 3: "width"},
            "image2": {0: "batch_size", 2: "height", 3: "width"},
            "optical_flow": {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    print(f"✅ Exported to {output_path}")

    # --- Step 1: Check ONNX Model Integrity ---
    print("\nVerifying ONNX graph structure...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model graph structure is valid!")

    # --- Step 2: Perform Dummy Inference with ONNX Runtime ---
    print("\nRunning dummy inference pass with ONNX Runtime...")
    session = ort.InferenceSession(
        output_path, 
        providers=["CPUExecutionProvider"]
    )

    # Generate dummy input numpy arrays with shape (1, 3, 360, 480)
    np_img1 = np.random.randn(1, 3, 360, 480).astype(np.float32)
    np_img2 = np.random.randn(1, 3, 360, 480).astype(np.float32)

    input_feed = {
        "image1": np_img1,
        "image2": np_img2,
    }

    outputs = session.run(["optical_flow"], input_feed)
    flow_output = outputs[0]

    print("✅ Inference pass successful!")
    print(f"   • Output shape : {flow_output.shape}")
    print(f"   • Output dtype : {flow_output.dtype}")
    print(f"   • Mean flow vector: {flow_output.mean():.4f}")


if __name__ == "__main__":
    export_and_validate_raft_onnx()