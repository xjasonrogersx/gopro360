import os
import sys
from pathlib import Path

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
	os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RAFT_EXAMPLE_DIR = ROOT / "ONNX-RAFT-Optical-Flow-Estimation"
if str(RAFT_EXAMPLE_DIR) not in sys.path:
	sys.path.insert(0, str(RAFT_EXAMPLE_DIR))

from raft import Raft

FLOW_FRAME_OFFSET = 10  # Number of frame difference to estimate the optical flow

# Initialize model
model_path = str(RAFT_EXAMPLE_DIR / "models" / "raft_things_iter20_480x640.onnx")
flow_estimator = Raft(model_path)

# Initialize video
cap = cv2.VideoCapture(str(ROOT / "GS010014-eqr-prores-telemetry.mov"))

# Skip first {start_time} seconds
start_time = 5
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)

show_window = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
if show_window:
	cv2.namedWindow("Estimated flow", cv2.WINDOW_NORMAL)

frame_list = []
frame_num = 0
while cap.isOpened():

	try:
		# Read frame from the video
		ret, prev_frame = cap.read()
		frame_list.append(prev_frame)
		if not ret:	
			break
	except:
		continue

	# Skip the first frames to be able to 
	frame_num += 1
	if frame_num <= FLOW_FRAME_OFFSET:
		continue

	flow_map = flow_estimator(frame_list[0], frame_list[-1])
	flow_img = flow_estimator.draw_flow()

	alpha = 0.5
	combined_img = cv2.addWeighted(frame_list[0], alpha, flow_img, (1-alpha),0)
	# combined_img = np.hstack((frame_list[-1], flow_img))

	if show_window:
		cv2.imshow("Estimated flow", combined_img)

	# Remove the oldest frame
	frame_list.pop(0)

	# Press key q to stop
	if show_window and cv2.waitKey(1) == ord('q'):
		break

print("Done processing video")	

cap.release()
cv2.destroyAllWindows()