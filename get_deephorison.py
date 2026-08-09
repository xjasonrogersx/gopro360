import torch
import torch.onnx
# Import your specific DeepHorizon network architecture
from deephorizon import DeepHorizonModel 

# 1. Initialize and load trained weights
model = DeepHorizonModel()
model.load_state_dict(torch.load("deephorizon_weights.pth", map_location="cpu"))
model.eval()

# 2. Create dummy input matching model expectations (e.g., 1 batch, 3 channels, 224x224)
# Adjust dimensions based on your specific DeepHorizon configuration
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Export to ONNX format
onnx_file_path = "deephorizon.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_file_path, 
    export_params=True, 
    opset_version=12,  # Opset 11 or higher is recommended
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Allows dynamic batching
)
print(f"Model successfully saved to {onnx_file_path}")
