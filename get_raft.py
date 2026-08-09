import torch
from core.raft import RAFT  # Ensure the official RAFT repo is in your path

def export_raft_light_onnx():
    # Instantiate RAFT with the 'small' flag
    args = type('Args', (object,), {
        'small': True, 
        'mixed_precision': False, 
        'alternate_corr': False
    })()
    
    model = RAFT(args)
    # Load your checkpoint weights if available:
    # model.load_state_dict(torch.load("raft-small.pth", map_dict='cpu'))
    model.eval()

    # Create dummy frame tensors with targeted sizing (e.g., 360x480)
    img1 = torch.randn(1, 3, 360, 480)
    img2 = torch.randn(1, 3, 360, 480)

    # Export using Opset 16+ to natively support modern operators 
    torch.onnx.export(
        model, 
        (img1, img2), 
        "raft_light_small_360x480.onnx",
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['image1', 'image2'],
        output_names=['optical_flow']
    )
    print("RAFT-Light successfully exported to ONNX format!")

export_raft_light_onnx()
