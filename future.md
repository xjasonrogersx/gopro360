


## RAFT-ONNX
https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation

to locate the motion center

```
import numpy as np
import onnxruntime as ort

# 1. Load the RAFT-Light ONNX model
model_path = "raft_light_small_360x480.onnx"
session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# 2. Prepare mock inputs (two sequential frames)
# Expected dimensions: [Batch, Channels, Height, Width]
# Image pixel values must usually be normalized between [-1, 1] or [0, 255] based on training
batch, channels, height, width = 1, 3, 360, 480
image1 = np.random.randn(batch, channels, height, width).astype(np.float32)
image2 = np.random.randn(batch, channels, height, width).astype(np.float32)

# 3. Match the expected input names of your specific ONNX export
inputs = {
    session.get_inputs()[0].name: image1,
    session.get_inputs()[1].name: image2
}

# 4. Run the inference pipeline
outputs = session.run(None, inputs)

# The output is a dense flow field vector map of shape [Batch, 2, Height, Width]
# Index 0 is horizontal displacement (Δx), Index 1 is vertical displacement (Δy)
flow_predictions = outputs[0] 
print("Predicted Flow Shape:", flow_predictions.shape)
```




## DeepHorizon

* Deep Horizon / Camera Pose Estimation ModelsIf your goal is to correct camera tilt ($pitch$) and roll ($roll$) relative to the true horizon so that your 360 video stays level:HorizonNet / LayoutNet (ONNX exported): Designed specifically for 360° equirectangular panoramas. While primarily used for 306 indoor room layout estimation, models like HorizonNet output 1D feature curves for the ceiling and floor boundaries, which can be adapted to find the horizontal level.

* Deep Horizon Line Detection (e.g., DeepHorizon / Work-Line models): Deep neural networks specifically trained to estimate the parameterized horizon line $(r, \theta)$ or camera orientation matrix directly from an image. You can export pretrained PyTorch weights for these architectures directly to .onnx.


```
import onnxruntime as ort
import numpy as np
import cv2

# 1. Load the ONNX model and start the inference session
session = ort.InferenceSession("deephorizon.onnx", providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])

# 2. Preprocess your image to match dummy_input dimensions
image = cv2.imread("test_scene.jpg")
image_resized = cv2.resize(image, (224, 224))
image_data = image_resized.astype(np.float32) / 255.0  # Normalize if required by your training pipeline
image_data = np.transpose(image_data, (2, 0, 1))       # Change to CHW format
input_tensor = np.expand_dims(image_data, axis=0)     # Add batch dimension (1, 3, 224, 224)

# 3. Execute inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
raw_predictions = session.run([output_name], {input_name: input_tensor})

# 4. Extract horizon line coordinates/parameters from output
horizon_output = raw_predictions[0]
print("DeepHorizon Predictions:", horizon_output)
```




---

update /home/jason/work/gopro360/viewer-gpmf.py
look at ### Extracting & Using GoPro Max Telemetry in Python in /home/jason/work/gopro360/plan.md and implement this in viewer-gpmf.py.
extract telemety using gpmf
Use CORI and GPS5

implemt Pitch & Roll (Leveling)
Yaw (Course Vectoring) with options for the 2 methods mentions.
On teh dislay draw the detected horisonal in yellow which moves as i pan.

when saving, apply the leveling and corse verctoring based on direct i chose.

update readme