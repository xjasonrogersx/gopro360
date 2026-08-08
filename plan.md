# Workflow Plan: 360 Video Levelling & Reframing

This plan outlines the end-to-end process for processing raw 360-degree equirectangular footage, leveling it using IMU telemetry, and applying smooth yaw transitions.

## Phase 1: Data Extraction
1.  **Extract Telemetry:** Use `exiftool` or `telemetry-parser` to extract gyro and accelerometer data from the `.MP4` or `.INSV` file.
2.  **Telemetry Cleanup:** Apply a low-pass filter (moving average) to the raw telemetry to remove sensor jitter.
3.  **Coordinate Mapping:** Align the extracted timestamps with video frames to ensure frame-accurate pitch/roll compensation.

## Phase 2: Processing Pipeline
1.  **Rotation Matrix Engine:** Implement the 3D spherical rotation engine using NumPy to convert Euler angles (yaw, pitch, roll) into a coordinate remapping grid.
2.  **Smooth Motion Logic:**
    *   **Yaw:** Use `scipy.interpolate.CubicSpline` to define keyframe-based panning for smooth camera transitions.
    *   **Pitch/Roll:** Invert the leveled telemetry data to counteract camera tilt.
3.  **Frame-by-Frame Processing:**
    *   Iterate through the video using `cv2.VideoCapture`.
    *   Apply `cv2.remap()` with `borderMode=cv2.BORDER_WRAP` to handle the spherical seam.
    *   Write the processed frames using `cv2.VideoWriter`.

## Phase 3: Quality Assurance
1.  **Visual Inspect:** Check for "wobble" (residual IMU noise) or "tearing" (seam artifacts).
2.  **Performance Check:** If processing time is high, consider using GPU-accelerated remapping with `cupy` or pre-calculating maps for steady sections of the video.

## References
- OpenCV (`cv2.remap`)
- NumPy (3D matrix operations)
- SciPy (CubicSpline interpolation)



# Equirectangular Projection Explained - Google Gemini


You said

What do you call a 360 video steatched to fill the rectangle

## Gemini said

An **equirectangular projection** (often referred to simply as an **equirectangular video** or **lat-long video**).

It maps the surface of a 360° sphere directly onto a flat rectangular frame—typically using a 2:1 aspect ratio (e.g., 3840×1920 or 5760×2880).

![Equirectangular mapping grid, AI generated](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRd6Y3hfUCZgWKD1Loe5yT6aJgS0nwDF-HuvcigZ4fAG4k686nNuLtSFUeH&s=10)

Equirectangular mapping grid. Source: Trek View

### Key Characteristics

-   **Distortion Pattern:** The horizontal center (the horizon) looks relatively normal, but the top (zenith) and bottom (nadir) stretch significantly across the entire frame.
    
-   **Coordinate Mapping:** Horizontal pixels represent 360∘ of longitude (from −180∘ to +180∘), and vertical pixels represent 180∘ of latitude (from +90∘ to −90∘).
    
-   **Standard Use:** It is the standard raw storage and playback format for 360° media used by platforms like YouTube, Facebook, and VR video players (which convert the flat video back into an interactive sphere in real time).
    

Want to know how to convert equirectangular video to standard rectilinear or cubemap formats?

Yes

You said

It this what insta 360 uses

## Gemini said

**Yes, but only after stitching.**

When you record video on an Insta360 camera, the capture and export workflow operates in two main phases:

### 1\. Raw Capture (Dual Fisheye)

The camera's raw output file (`.INSV`) doesn't start as an equirectangular image. Instead, it records **two side-by-side circular fisheye images**—one from the front lens and one from the back lens.

### 2\. Export (Equirectangular or Reframed)

When you export 360° video out of **Insta360 Studio** or the **Insta360 App**, the software stitches those two fisheye images together and stretches them into a standard **equirectangular video** (usually as a standard `.MP4` file).

Niantic Spatial

This equirectangular MP4 is what you upload if you want interactive 360° playback on YouTube, Facebook, or VR headsets.

### What Insta360 typically exports:

Export Goal

Format

Result

**Interactive 360° Video**

Equirectangular MP4

Stretched flat video with embedded 360 metadata, ready for VR viewers or YouTube.

**Standard "Flat" Video**

Reframed / Rectilinear MP4

A normal video file where you manually control keyframes, pan/zoom, or track subjects.

You said

Ok, I have a equirectangular MP4 video. Can I dewarp this in opencv

## Gemini said

Yes. You do this by performing a **3D perspective transformation** (converting spherical coordinates back into a rectilinear image).

Instead of iterating pixel-by-pixel in Python—which is extremely slow for video—you generate a 2D lookup map once using **NumPy** and pass it to **`cv2.remap()`**, which handles pixel interpolation on the C++ layer in milliseconds.

### Working Python Implementation

This script extracts a normal field-of-view (FOV) perspective frame from an equirectangular video file.

Python

```
import cv2
import numpy as np

def build_equirectangular_map(out_hw, fov_deg, yaw_deg, pitch_deg, in_hw):
    """
    Generates x and y pixel mapping grids for cv2.remap().
    """
    out_h, out_w = out_hw
    in_h, in_w = in_hw
    
    # 1. Create normalized 2D grid for the target perspective frame
    fov_rad = np.radians(fov_deg)
    f = 0.5 * out_w / np.tan(0.5 * fov_rad)  # Focal length
    
    x = np.arange(out_w) - (out_w / 2.0)
    y = np.arange(out_h) - (out_h / 2.0)
    xx, yy = np.meshgrid(x, y)
    
    # 3D points on the virtual image plane (Z points forward)
    pts_3d = np.stack([xx, yy, np.full_like(xx, f)], axis=-1)
    
    # Normalize vectors to unit length sphere
    norm = np.linalg.norm(pts_3d, axis=-1, keepdims=True)
    pts_3d = pts_3d / norm

    # 2. Rotate camera view (Yaw & Pitch)
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # Rotation matrices
    R_yaw = np.array([
        [np.cos(yaw),  0, np.sin(yaw)],
        [0,            1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    R_pitch = np.array([
        [1, 0,             0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])
    
    # Apply rotation
    R = R_yaw @ R_pitch
    pts_3d_rot = pts_3d @ R.T

    # 3. Convert rotated 3D cartesian coordinates to spherical angles
    x_rot = pts_3d_rot[..., 0]
    y_rot = pts_3d_rot[..., 1]
    z_rot = pts_3d_rot[..., 2]

    longitude = np.arctan2(x_rot, z_rot)          # -pi to pi
    latitude = np.arcsin(np.clip(-y_rot, -1, 1))  # -pi/2 to pi/2

    # 4. Map spherical angles to Equirectangular pixel positions
    map_x = ((longitude / (2 * np.pi)) + 0.5) * in_w
    map_y = ((0.5 - (latitude / np.pi))) * in_h

    return map_x.astype(np.float32), map_y.astype(np.float32)

# --- Example Pipeline Usage ---
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Desired perspective video dimensions & camera orientation
    out_w, out_h = 1280, 720
    fov = 90      # Field of view in degrees
    yaw = 0       # Pan angle (-180 to 180)
    pitch = 0     # Tilt angle (-90 to 90)

    # Pre-calculate mapping coordinates (only do this ONCE for speed)
    map_x, map_y = build_equirectangular_map(
        (out_h, out_w), fov, yaw, pitch, (in_h, in_w)
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply dewarping map to frame
        dewarped_frame = cv2.remap(
            frame, map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_WRAP
        )

        out.write(dewarped_frame)

    cap.release()
    out.release()

# Run the dewarp function
process_video("equirectangular_input.mp4", "normal_output.mp4")
```

### How it works:

1.  **Grid Generation:** It builds a 3D ray-tracing grid corresponding to a traditional camera lens looking straight ahead.
    
2.  **Rotation:** Modifying `yaw` and `pitch` rotates these rays in 3D space to point the virtual camera anywhere inside the 360 sphere.
    
    Codingballad
    
3.  **Spherical Projection:** It converts these rotated 3D vectors into (x,y) coordinate lookups on your original equirectangular frame.
    
4.  **Fast Remapping:** `cv2.remap()` pulls pixels directly from the original image based on that lookup table. Since the map calculation is performed only once outside the main video loop, execution runs at near native speed.
    

You said

Can I detect the horizon, is there a onnx model that can do this

## Gemini said

Yes, you can detect the horizon in an equirectangular video, and using an **ONNX model** is one of the most efficient ways to do it.

However, how you approach it depends on whether you want to detect **structural features** (like sea/sky or terrain boundaries) or if you are looking to **stabilize/level camera tilt and roll** using visual horizon cues.

### Recommended ONNX Models & Approaches

#### 1\. Semantic Segmentation Models (Best for Outdoor / Marine Environments)

If your goal is to locate the boundary between the ground/water and the sky, **semantic segmentation** models pre-trained on sky/ground categories work exceptionally well.

-   **Segment Anything Model (SAM / MobileSAM ONNX):** You can run lightweight exported ONNX variants like **MobileSAM** or **FastSAM**. By prompting or querying the `sky` or `ground` mask, you can find the contour edge corresponding to the visual horizon line.
    
-   **DeepLabV3+ or SegFormer (ONNX):** Standard lightweight vision backbones (e.g., SegFormer-B0 exported to ONNX) trained on Cityscapes or ADE20K explicitly output a `sky` class mask. Finding the bottom boundary of the sky mask gives you pixel coordinates for the horizon across the frame.
    

#### 2\. Deep Horizon / Camera Pose Estimation Models

If your goal is to correct camera tilt (pitch) and roll (roll) relative to the true horizon so that your 360 video stays level:

-   **HorizonNet / LayoutNet (ONNX exported):** Designed specifically for 360° equirectangular panoramas. While primarily used for 306 indoor room layout estimation, models like **HorizonNet** output 1D feature curves for the ceiling and floor boundaries, which can be adapted to find the horizontal level.
    
-   **Deep Horizon Line Detection (e.g., DeepHorizon / Work-Line models):** Deep neural networks specifically trained to estimate the parameterized horizon line (r,θ) or camera orientation matrix directly from an image. You can export pretrained PyTorch weights for these architectures directly to `.onnx`.
    

### Alternative: Classical OpenCV Method (Fast & No GPU Needed)

If your equirectangular video has a relatively clear visual boundary (e.g., kayaking on water, outdoor landscapes), you can often detect the horizon line in a few lines of OpenCV code without deep learning overhead:

Python

```
import cv2
import numpy as np

def detect_horizon_classical(frame):
    # 1. Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Vertical gradient (Sobel Y) to highlight horizontal lines
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_y = np.uint8(np.absolute(sobel_y))
    
    # 3. Threshold edge map
    _, thresh = cv2.threshold(abs_sobel_y, 50, 255, cv2.THRESH_BINARY)
    
    # 4. Find dominant horizontal line across columns
    # Sum across columns to find peak row per column region
    row_sums = np.sum(thresh, axis=1)
    estimated_horizon_y = np.argmax(row_sums)
    
    return estimated_horizon_y
```

### Key Consideration for Equirectangular Format

In a perfectly level equirectangular image, the true horizon is **always a straight line exactly at the vertical midpoint** (y\=height/2).

If the camera tilts or rolls:

-   **Pitch (tilting up/down):** Moves the horizon line vertically up or down.
    
-   **Roll (leaning side-to-side):** Turns the horizon into a **sine wave** across the 2:1 rectangular frame.
    

If you are using an ONNX segmentation model to detect the sine-wave contour of a rolled horizon, you can fit a sine curve y(θ)\=Asin(θ+ϕ)+C to the detected boundary points. The amplitude A and phase ϕ directly tell you the camera's exact **roll** and **pitch** angles required to level the video.

Would you like an example script running MobileSAM/SegFormer ONNX for segmentation or a curve-fitting script to auto-level 360 video pitch and roll?

Yes

You said

I'm on a river so horizon is not obvious

## Gemini said

When you are down on a river—surrounded by trees, riverbanks, canyon walls, or water spray—a traditional "sky meets flat ground" visual horizon detector won't work because the real horizon line is almost entirely blocked.

However, you can still find the **true gravity horizon** (or auto-level your footage) using one of three effective strategies:

### Strategy 1: Use the IMU / Telemetry Data (Simplest & Most Accurate)

If your equirectangular video was recorded on an action camera like an **Insta360** (One X, X2, X3, X4) or a **GoPro Max**, the camera records built-in **gyroscope and accelerometer data** directly into the file metadata or an accompanying telemetry stream.

-   **How it works:** The internal IMU measures gravity's vector relative to the camera frame for every millisecond of recording, regardless of visual occlusions like trees or banks.
    
-   **How to extract it:** You can extract this orientation matrix (pitch, roll, yaw) per frame using open-source tools like **`gpmf-parser`** (GoPro) or **`telemetry-parser` / `exiftool`** (Insta360) in Python.
    
-   **Applying it:** Pass these exact pitch and roll angles straight into the 3D rotation matrix (Rpitch​⋅Rroll​) inside your `cv2.remap()` script to level the horizon automatically without relying on any visual detection.
    

### Strategy 2: Vanishing Point & Optical Flow (Deep Learning)

When trees or riverbanks obscure the line where the sky meets the earth, human eyes still know what "level" looks like because trees grow vertically and water flows horizontally.

You can use **Optical Flow** or **Vanishing Point Detection ONNX models**:

1.  **Optical Flow (Motion Vectoring):** As you move down the river, distant objects expand outward from a single point—the **Focus of Expansion (FOE)**. Running an ONNX optical flow model (like **RAFT-Light** or **PNCFlow** in ONNX format) on consecutive frames identifies the direction of travel. The FOE corresponds directly to your forward heading vector on the equirectangular sphere.
    
2.  **Vertical Structural Cues:** Deep learning models trained for **Upright Orientation / Vanishing Point Detection** (e.g., ONNX exports of models like _DeepVP_ or _LayoutNet_) detect dominant vertical lines (tree trunks, cliff edges). Because gravity dictates that trees grow upward, these vertical lines converge toward the zenith (top pole), allowing you to infer roll and pitch even when the ground plane is completely hidden.
    

### Strategy 3: Water Surface Geometry & Edge Fitting

Even on a river, the water surface directly around your boat forms a locally flat horizontal plane.

Python

```
import cv2
import numpy as np

def estimate_water_plane_roll(frame):
    """
    Estimates roll angle by finding the tilt of local water-to-bank boundary lines.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Focus on the lower half of the equirectangular frame where the river is
    h, w = gray.shape
    roi = gray[int(h*0.4):int(h*0.8), :]
    
    # Detect prominent edges along the riverbank boundary
    edges = cv2.Canny(roi, 50, 150)
    
    # Use Hough Line Transform to detect long structural lines along the water's edge
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=w//10, maxLineGap=20)
    
    if lines is me:
        return 0.0
        
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate slope angle of lines near the water level
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Filter for near-horizontal lines
        if abs(angle) < 30:
            angles.append(angle)
            
    if not angles:
        return 0.0
        
    # Median angle gives an estimate of camera roll relative to the river plane
    return float(np.median(angles))
```

### Summary Recommendation

Goal

Best Method

**Highest Precision**

Extract built-in **IMU telemetry** from the `.mp4`/`.insv` metadata and apply it directly to the transformation matrix.

**No Telemetry Available**

Run an **Optical Flow ONNX model** (e.g., RAFT-ONNX) to locate the motion center, or use vertical line vanishing points from tree trunks/cliffs.

Would you like a Python script to extract telemetry/gyro data directly from your video file?

Yes

You said

I've I have this info, is it possible to level (pitch and roll) and smoothly change the yaw

## Gemini said

**Yes, absolutely.** In fact, modifying the 3D rotation matrix per frame allows you to **level the camera (undoing unwanted pitch and roll)** while simultaneously **rotating the virtual view direction (animating yaw smoothly)**.

GitHub

Because an equirectangular video maps a complete 360∘×180∘ sphere onto a 2D plane, rotating the sphere in 3D space and re-projecting it back onto the equirectangular grid yields a perfectly leveled, re-oriented 360 video.

### How the Math Works

To accomplish both tasks at the same time for every frame:

1.  **Pitch & Roll Correction:** Apply the inverse of the IMU's pitch and roll (−θpitch​,−θroll​) to eliminate camera tilt.
    
2.  **Yaw Interpolation:** Pass a smoothly changing target yaw value ψ(t) into the rotation matrix.
    
3.  **Combined Transformation:** Combine all three into a single 3D rotation matrix:
    
    R(t)\=Ryaw​(ψ(t))⋅Rpitch​(−θpitch​)⋅Rroll​(−θroll​)
    

### Python Implementation (Equirectangular-to-Equirectangular)

This script processes an equirectangular video frame-by-frame, applying per-frame IMU levelling and dynamically panning the yaw smoothly over time.

Python

```
import cv2
import numpy as np

def build_equirect_rotation_map(in_hw, yaw_deg, pitch_deg, roll_deg):
    """
    Computes a 2D remapping grid (map_x, map_y) to rotate an 
    equirectangular frame in 3D space.
    """
    h, w = in_hw
    
    # 1. Generate 2D meshgrid of lat/long angles for destination image
    x = (np.arange(w) / w - 0.5) * (2 * np.pi)       # Longitude: -pi to pi
    y = (0.5 - np.arange(h) / h) * np.pi              # Latitude:  pi/2 to -pi/2
    xx, yy = np.meshgrid(x, y)
    
    # 2. Convert destination grid to 3D Cartesian points on a unit sphere
    X = np.cos(yy) * np.sin(xx)
    Y = np.sin(yy)
    Z = np.cos(yy) * np.cos(xx)
    pts_3d = np.stack([X, Y, Z], axis=-1)  # Shape: (h, w, 3)

    # 3. Build 3D Euler Rotation Matrix (Yaw, Pitch, Roll)
    rad_y = np.radians(yaw_deg)
    rad_p = np.radians(pitch_deg)
    rad_r = np.radians(roll_deg)

    # Rotation Z (Yaw)
    R_yaw = np.array([
        [np.cos(rad_y), -np.sin(rad_y), 0],
        [np.sin(rad_y),  np.cos(rad_y), 0],
        [0,              0,             1]
    ])
    # Rotation X (Pitch)
    R_pitch = np.array([
        [1, 0,              0],
        [0, np.cos(rad_p), -np.sin(rad_p)],
        [0, np.sin(rad_p),  np.cos(rad_p)]
    ])
    # Rotation Y (Roll)
    R_roll = np.array([
        [np.cos(rad_r), 0, np.sin(rad_r)],
        [0,             1, 0],
        [-np.sin(rad_r),0, np.cos(rad_r)]
    ])

    # Combined Rotation Matrix
    R = R_yaw @ R_pitch @ R_roll

    # 4. Rotate 3D unit sphere vectors
    pts_3d_rot = pts_3d @ R.T

    # 5. Map rotated 3D points back to spherical coordinates
    x_rot = pts_3d_rot[..., 0]
    y_rot = pts_3d_rot[..., 1]
    z_rot = pts_3d_rot[..., 2]

    long_rot = np.arctan2(x_rot, z_rot)
    lat_rot = np.arcsin(np.clip(y_rot, -1.0, 1.0))

    # 6. Convert spherical angles back to source image pixel coordinates
    map_x = ((long_rot / (2 * np.pi)) + 0.5) * w
    map_y = (0.5 - (lat_rot / np.pi)) * h

    return map_x.astype(np.float32), map_y.astype(np.float32)

def process_and_level_video(input_path, output_path, imu_pitch_list, imu_roll_list):
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get current frame's IMU pitch and roll correction
        pitch_corr = -imu_pitch_list[frame_idx]  # Invert to cancel out tilt
        roll_corr  = -imu_roll_list[frame_idx]   # Invert to cancel out roll

        # Smooth Yaw Paning: e.g., pan 360 degrees smoothly across the entire video
        progress = frame_idx / max(total_frames, 1)
        yaw_pan = progress * 360.0  # Smoothly pans forward -> right -> back -> left

        # Calculate map for current frame orientation
        map_x, map_y = build_equirect_rotation_map((h, w), yaw_pan, pitch_corr, roll_corr)

        # Apply transformation with border wrapping for seamless 360 stitching
        leveled_frame = cv2.remap(
            frame, map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_WRAP
        )

        out.write(leveled_frame)
        frame_idx += 1

    cap.release()
    out.release()
```

### Tips for Smooth Yaw Transitions

1.  **Smoothing Raw IMU Data:** Accelerometer/gyro data can contain high-frequency noise. Apply a **moving average** or **low-pass filter** (using `scipy.ndimage.gaussian_filter1d`) to your pitch and roll arrays before passing them into the rotation matrix to avoid jittery output.
    
2.  **Keyframed Yaw Panning:** Rather than a simple linear rotation (0∘→360∘), you can define keyframes (e.g., look ahead at river bend, smoothly pan left toward a rapid) and use **cubic spline interpolation** (`scipy.interpolate.CubicSpline`) to generate smooth yaw curves over time.
    

You said

Make a plan in md

Show code

## Gemini said

Python

```
markdown_content = """# Workflow Plan: 360 Video Levelling & Reframing

This plan outlines the end-to-end process for processing raw 360-degree equirectangular footage, leveling it using IMU telemetry, and applying smooth yaw transitions.

## Phase 1: Data Extraction
1.  **Extract Telemetry:** Use `exiftool` or `telemetry-parser` to extract gyro and accelerometer data from the `.MP4` or `.INSV` file.
2.  **Telemetry Cleanup:** Apply a low-pass filter (moving average) to the raw telemetry to remove sensor jitter.
3.  **Coordinate Mapping:** Align the extracted timestamps with video frames to ensure frame-accurate pitch/roll compensation.

## Phase 2: Processing Pipeline
1.  **Rotation Matrix Engine:** Implement the 3D spherical rotation engine using NumPy to convert Euler angles (yaw, pitch, roll) into a coordinate remapping grid.
2.  **Smooth Motion Logic:**
    *   **Yaw:** Use `scipy.interpolate.CubicSpline` to define keyframe-based panning for smooth camera transitions.
    *   **Pitch/Roll:** Invert the leveled telemetry data to counteract camera tilt.
3.  **Frame-by-Frame Processing:**
    *   Iterate through the video using `cv2.VideoCapture`.
    *   Apply `cv2.remap()` with `borderMode=cv2.BORDER_WRAP` to handle the spherical seam.
    *   Write the processed frames using `cv2.VideoWriter`.

## Phase 3: Quality Assurance
1.  **Visual Inspect:** Check for "wobble" (residual IMU noise) or "tearing" (seam artifacts).
2.  **Performance Check:** If processing time is high, consider using GPU-accelerated remapping with `cupy` or pre-calculating maps for steady sections of the video.

## References
- OpenCV (`cv2.remap`)
- NumPy (3D matrix operations)
- SciPy (CubicSpline interpolation)
"""

with open("360_video_workflow.md", "w") as f:
    f.write(markdown_content)

```
