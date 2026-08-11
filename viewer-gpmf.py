from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation as R

RAFT_EXAMPLE_DIR = Path("/home/jason/work/gopro360/ONNX-RAFT-Optical-Flow-Estimation")
if str(RAFT_EXAMPLE_DIR) not in sys.path:
	sys.path.insert(0, str(RAFT_EXAMPLE_DIR))

from raft import Raft


WINDOW_NAME = "gopro360 viewer"
RAFT_PREVIEW_WINDOW = "RAFT crop preview"
MIN_FOV = 30.0
MAX_FOV = 170.0
ASPECT_PRESETS = [
	("1:1", 1080, 1080),
	("16:9", 1920, 1080),
	("9:16", 1080, 1920),
	("4:3", 1440, 1080),
]
DEFAULT_RAFT_ONNX = Path("/home/jason/work/gopro360/ONNX-RAFT-Optical-Flow-Estimation/models/raft_small_iter20_360x480.onnx")


@dataclass
class ExportStatus:
	active: bool = False
	done: bool = False
	success: bool = False
	message: str = ""
	output_path: str = ""
	processed: int = 0
	total: int = 0
	started_at: float = 0.0


@dataclass
class TelemetryData:
	pitch_deg: np.ndarray | None = None
	roll_deg: np.ndarray | None = None
	gps_heading_deg: np.ndarray | None = None
	flow_heading_deg: np.ndarray | None = None
	grav: np.ndarray | None = None
	grav_timestamps: np.ndarray | None = None
	cori: np.ndarray | None = None
	cori_timestamps: np.ndarray | None = None
	shut: np.ndarray | None = None
	shut_timestamps: np.ndarray | None = None
	isoe: np.ndarray | None = None
	isoe_timestamps: np.ndarray | None = None
	source: str = "none"


@dataclass
class ViewerState:
	yaw: float
	pitch: float
	roll: float
	fov: float
	preset_idx: int
	paused: bool = True
	show_info: bool = True
	dragging: bool = False
	last_mouse_x: int = 0
	last_mouse_y: int = 0
	dirty: bool = True
	frame_idx: int = 0
	status_text: str = ""
	status_until: float = 0.0
	map_key: tuple | None = None
	map_x: np.ndarray | None = None
	map_y: np.ndarray | None = None
	telemetry: TelemetryData = field(default_factory=TelemetryData)
	use_auto_level: bool = True
	yaw_mode: str = "manual"
	raft_active: bool = False
	raft_status: str = ""
	raft_input_crop: np.ndarray | None = None
	raft_output: np.ndarray | None = None
	raft_thread: threading.Thread | None = None
	export_lock: threading.Lock = field(default_factory=threading.Lock)
	export_status: ExportStatus = field(default_factory=ExportStatus)
	export_thread: threading.Thread | None = None


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Interactive 360 equirectangular OpenCV viewer")
	parser.add_argument("input_video", help="Path to equirectangular input video")
	parser.add_argument(
		"--output",
		default="",
		help="Output path for `s` export. Defaults to <input_stem>_dewarped.mp4",
	)
	parser.add_argument("--yaw", type=float, default=0.0, help="Initial yaw in degrees")
	parser.add_argument("--pitch", type=float, default=0.0, help="Initial pitch in degrees")
	parser.add_argument("--roll", type=float, default=0.0, help="Initial roll in degrees")
	parser.add_argument("--fov", type=float, default=90.0, help="Initial horizontal FOV in degrees")
	parser.add_argument(
		"--preset",
		type=int,
		default=1,
		help="Initial aspect preset index: 0=1:1, 1=16:9, 2=9:16, 3=4:3",
	)
	parser.add_argument(
		"--raft-onnx",
		default=str(DEFAULT_RAFT_ONNX),
		help="Path to RAFT ONNX model used for optical-flow yaw when GPS heading is unavailable",
	)
	return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
	return max(low, min(high, value))


def round_even(value: float) -> int:
	rounded = int(round(value))
	return rounded if rounded % 2 == 0 else rounded - 1


def wrap_angle(angle_deg: float) -> float:
	wrapped = (angle_deg + 180.0) % 360.0 - 180.0
	return wrapped


def _coerce_array(payload: object) -> np.ndarray | None:
	if payload is None:
		return None
	try:
		array = np.asarray(payload, dtype=np.float32)
	except Exception:
		return None
	if array.size == 0:
		return None
	return array


def _normalize_stream_samples(payload: object) -> np.ndarray | None:
	array = _coerce_array(payload)
	if array is None:
		return None
	if array.ndim == 1:
		return array.reshape(-1, 1)
	if array.ndim == 2:
		return array
	return array.reshape(array.shape[0], -1)


def _normalize_timestamps(payload: object, expected_len: int) -> np.ndarray | None:
	array = _coerce_array(payload)
	if array is None:
		return None
	timestamps = array.reshape(-1).astype(np.float32)
	if timestamps.size == expected_len:
		return timestamps
	if timestamps.size == 1 and expected_len > 1:
		return np.linspace(float(timestamps[0]), float(timestamps[0]), expected_len, dtype=np.float32)
	if timestamps.size > 1 and expected_len > 1:
		source_t = np.linspace(0.0, 1.0, timestamps.size, dtype=np.float32)
		target_t = np.linspace(0.0, 1.0, expected_len, dtype=np.float32)
		return np.interp(target_t, source_t, timestamps).astype(np.float32)
	return None


def interpolate_series(values: np.ndarray | None, total_frames: int) -> np.ndarray | None:
	if values is None or values.size == 0:
		return None
	values = np.asarray(values, dtype=np.float32)
	if total_frames <= 0:
		return values[:1].copy()
	if values.size == 1:
		return np.full(total_frames, values[0], dtype=np.float32)
	source_t = np.linspace(0.0, max(total_frames - 1, 1), values.size)
	frame_t = np.arange(total_frames, dtype=np.float32)
	interp = interp1d(source_t, values, kind="linear", bounds_error=False, fill_value=(values[0], values[-1]))
	return interp(frame_t).astype(np.float32)


def load_gpmf_telemetry(video_path: Path, total_frames: int) -> TelemetryData:
	telemetry = TelemetryData()
	source_parts: list[str] = []

	extractor = GoProTelemetryExtractor(str(video_path))
	try:
		extractor.open_source()
		for key in ("GRAV", "CORI", "SHUT", "ISOE"):
			try:
				stream_data, timestamps = extractor.extract_data(key)
			except Exception:
				continue

			samples = _normalize_stream_samples(stream_data)
			if samples is None or samples.shape[0] == 0:
				continue

			ts = _normalize_timestamps(timestamps, samples.shape[0])
			if key == "GRAV":
				telemetry.grav = samples.astype(np.float32)
				telemetry.grav_timestamps = ts
				source_parts.append("grav")
			elif key == "CORI":
				telemetry.cori = samples.astype(np.float32)
				telemetry.cori_timestamps = ts
				source_parts.append("cori")

				quats = telemetry.cori
				if quats is not None and quats.size > 0:
					if quats.shape[1] != 4 and quats.shape[0] == 4:
						quats = quats.T
					if quats.shape[1] == 4:
						# GoPro CORI is commonly w,x,y,z while SciPy expects x,y,z,w.
						if float(np.mean(np.abs(quats[:, 0]))) > float(np.mean(np.abs(quats[:, 3]))):
							quats_xyzw = np.column_stack((quats[:, 1], quats[:, 2], quats[:, 3], quats[:, 0]))
						else:
							quats_xyzw = quats
						rot = R.from_quat(quats_xyzw)
						eulers = rot.as_euler("xyz", degrees=True)
						telemetry.pitch_deg = interpolate_series(eulers[:, 0], total_frames)
						telemetry.roll_deg = interpolate_series(eulers[:, 2], total_frames)
			elif key == "SHUT":
				telemetry.shut = samples.astype(np.float32)
				telemetry.shut_timestamps = ts
				source_parts.append("shut")
			elif key == "ISOE":
				telemetry.isoe = samples.astype(np.float32)
				telemetry.isoe_timestamps = ts
				source_parts.append("isoe")
	finally:
		try:
			extractor.close_source()
		except Exception:
			pass

	telemetry.source = "+".join(dict.fromkeys(source_parts)) if source_parts else "none"
	return telemetry


def build_raft_blob(frame: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
	resized = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_AREA)
	rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
	blob = np.transpose(rgb.astype(np.float32), (2, 0, 1))[None, ...]
	return blob


def extract_flow_x(flow_output: np.ndarray) -> np.ndarray | None:
	if flow_output.ndim == 4:
		flow = flow_output[0]
		if flow.shape[0] == 2:
			return flow[0]
	if flow_output.ndim == 3:
		if flow_output.shape[0] == 2:
			return flow_output[0]
		if flow_output.shape[-1] == 2:
			return flow_output[..., 0]
	if flow_output.ndim == 2:
		return flow_output
	return None


def compute_flow_heading_series(video_path: Path, total_frames: int, raft_onnx_path: Path) -> np.ndarray | None:
	if total_frames < 2 or not raft_onnx_path.exists():
		return None

	try:
		net = cv2.dnn.readNetFromONNX(str(raft_onnx_path))
	except Exception:
		return None

	input_h, input_w = 360, 480
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		return None

	print(f"Computing RAFT flow yaw ({total_frames} frames)...")

	ok, prev_frame = cap.read()
	if not ok:
		cap.release()
		return None

	frame_indices: list[int] = [0]
	yaw_points: list[float] = [0.0]
	cumulative_yaw = 0.0
	last_progress_print = 0.0

	for frame_idx in range(1, total_frames):
		ok, frame = cap.read()
		if not ok:
			break

		now = time.time()
		if frame_idx == 1 or now - last_progress_print >= 0.25 or frame_idx == total_frames - 1:
			pct = 100.0 * (frame_idx / max(total_frames - 1, 1))
			print(f"\rRAFT flow yaw: {frame_idx}/{total_frames - 1} ({pct:5.1f}%)", end="", flush=True)
			last_progress_print = now

		blob1 = build_raft_blob(prev_frame, input_h, input_w)
		blob2 = build_raft_blob(frame, input_h, input_w)

		try:
			net.setInput(blob1, "image1")
			net.setInput(blob2, "image2")
			flow_output = net.forward("optical_flow")
		except Exception:
			try:
				net.setInput(blob1)
				flow_output = net.forward()
			except Exception:
				prev_frame = frame
				continue

		flow_x = extract_flow_x(flow_output)
		if flow_x is None or flow_x.size == 0:
			prev_frame = frame
			continue

		h, w = flow_x.shape[:2]
		center_band = flow_x[h // 3 : (2 * h) // 3, w // 6 : (5 * w) // 6]
		if center_band.size == 0:
			prev_frame = frame
			continue

		dx = float(np.median(center_band))
		delta_yaw = clamp(-dx * (360.0 / float(input_w)), -8.0, 8.0)
		cumulative_yaw += delta_yaw
		frame_indices.append(frame_idx)
		yaw_points.append(cumulative_yaw)
		prev_frame = frame

	cap.release()
	print()

	if len(frame_indices) < 2:
		print("RAFT flow yaw failed: insufficient valid flow samples")
		return None

	x = np.asarray(frame_indices, dtype=np.float32)
	y = np.asarray(yaw_points, dtype=np.float32)
	target_x = np.arange(total_frames, dtype=np.float32)

	if len(frame_indices) >= 4:
		spline = CubicSpline(x, y, bc_type="natural")
		smoothed = spline(target_x).astype(np.float32)
	else:
		smoothed = np.interp(target_x, x, y).astype(np.float32)

	print(f"RAFT flow yaw ready: {len(frame_indices)} sampled frames")

	return smoothed


def sample_series(values: np.ndarray | None, frame_idx: int) -> float:
	if values is None or values.size == 0:
		return 0.0
	idx = min(max(frame_idx, 0), values.size - 1)
	return float(values[idx])


def sample_stream_at_time(values: np.ndarray | None, timestamps: np.ndarray | None, t_sec: float) -> np.ndarray | None:
	if values is None or values.size == 0:
		return None
	if timestamps is None or timestamps.size == 0:
		idx = min(max(int(round(t_sec)), 0), values.shape[0] - 1)
		return values[idx]

	idx = int(np.searchsorted(timestamps, t_sec, side="left"))
	if idx <= 0:
		return values[0]
	if idx >= timestamps.size:
		return values[-1]

	prev_idx = idx - 1
	if abs(float(timestamps[idx]) - t_sec) < abs(float(timestamps[prev_idx]) - t_sec):
		return values[idx]
	return values[prev_idx]


def format_sample(values: np.ndarray | None, decimals: int = 4) -> str:
	if values is None or values.size == 0:
		return "n/a"
	return np.array2string(values.astype(np.float32), precision=decimals, separator=",", suppress_small=False)


def get_effective_orientation(state: "ViewerState", frame_idx: int) -> tuple[float, float, float]:
	yaw_deg = state.yaw
	pitch_deg = state.pitch
	roll_deg = state.roll

	if state.use_auto_level:
		pitch_deg = state.pitch - sample_series(state.telemetry.pitch_deg, frame_idx)
		roll_deg = state.roll - sample_series(state.telemetry.roll_deg, frame_idx)

	if state.yaw_mode == "gps":
		yaw_deg = state.yaw + sample_series(state.telemetry.gps_heading_deg, frame_idx)
	elif state.yaw_mode == "flow":
		yaw_deg = state.yaw + sample_series(state.telemetry.flow_heading_deg, frame_idx)

	return yaw_deg, pitch_deg, roll_deg


def available_yaw_modes(telemetry: TelemetryData) -> list[str]:
	modes = ["manual"]
	if telemetry.gps_heading_deg is not None and telemetry.gps_heading_deg.size > 0:
		modes.append("gps")
	if telemetry.flow_heading_deg is not None and telemetry.flow_heading_deg.size > 0:
		modes.append("flow")
	return modes


def yaw_mode_label(yaw_mode: str) -> str:
	if yaw_mode == "gps":
		return "yaw=gps"
	if yaw_mode == "flow":
		return "yaw=flow"
	return "yaw=manual"


def build_remap(
	in_h: int,
	in_w: int,
	out_h: int,
	out_w: int,
	yaw_deg: float,
	pitch_deg: float,
	roll_deg: float,
	fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
	fov_rad = math.radians(clamp(fov_deg, 1.0, 179.0))
	focal = 0.5 * out_w / math.tan(0.5 * fov_rad)

	x = np.arange(out_w, dtype=np.float32) - (out_w * 0.5)
	y = np.arange(out_h, dtype=np.float32) - (out_h * 0.5)
	xx, yy = np.meshgrid(x, y)

	# Camera space: +Z forward, +X right, +Y up.
	rays = np.stack([xx, -yy, np.full_like(xx, focal)], axis=-1)
	rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

	yaw = math.radians(yaw_deg)
	pitch = math.radians(pitch_deg)
	roll = math.radians(roll_deg)

	r_yaw = np.array(
		[
			[math.cos(yaw), 0.0, math.sin(yaw)],
			[0.0, 1.0, 0.0],
			[-math.sin(yaw), 0.0, math.cos(yaw)],
		],
		dtype=np.float32,
	)
	r_pitch = np.array(
		[
			[1.0, 0.0, 0.0],
			[0.0, math.cos(pitch), -math.sin(pitch)],
			[0.0, math.sin(pitch), math.cos(pitch)],
		],
		dtype=np.float32,
	)
	r_roll = np.array(
		[
			[math.cos(roll), -math.sin(roll), 0.0],
			[math.sin(roll), math.cos(roll), 0.0],
			[0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)

	rotation = r_yaw @ r_pitch @ r_roll
	rays_rot = rays @ rotation.T

	x_rot = rays_rot[..., 0]
	y_rot = rays_rot[..., 1]
	z_rot = rays_rot[..., 2]

	lon = np.arctan2(x_rot, z_rot)
	lat = np.arcsin(np.clip(y_rot, -1.0, 1.0))

	map_x = ((lon / (2.0 * np.pi)) + 0.5) * in_w
	map_y = (0.5 - (lat / np.pi)) * in_h

	return map_x.astype(np.float32), map_y.astype(np.float32)


def get_cached_remap(state: ViewerState, in_h: int, in_w: int, out_h: int, out_w: int, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
	yaw_deg, pitch_deg, roll_deg = get_effective_orientation(state, frame_idx)
	key = (
		in_h,
		in_w,
		out_h,
		out_w,
		round(yaw_deg, 4),
		round(pitch_deg, 4),
		round(roll_deg, 4),
		round(state.fov, 4),
		frame_idx,
		state.use_auto_level,
		state.yaw_mode,
	)
	if state.map_key != key or state.map_x is None or state.map_y is None:
		state.map_x, state.map_y = build_remap(
			in_h=in_h,
			in_w=in_w,
			out_h=out_h,
			out_w=out_w,
			yaw_deg=yaw_deg,
			pitch_deg=pitch_deg,
			roll_deg=roll_deg,
			fov_deg=state.fov,
		)
		state.map_key = key
	return state.map_x, state.map_y


def render_frame(frame: np.ndarray, state: ViewerState) -> np.ndarray:
	_, out_w, out_h = ASPECT_PRESETS[state.preset_idx]
	in_h, in_w = frame.shape[:2]
	map_x, map_y = get_cached_remap(state, in_h, in_w, out_h, out_w, state.frame_idx)
	return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def draw_horizon_overlay(frame: np.ndarray, state: ViewerState) -> None:
	h, w = frame.shape[:2]
	yaw_deg, pitch_deg, roll_deg = get_effective_orientation(state, state.frame_idx)
	horizon_y = (h / 2.0) - (pitch_deg * (h / 180.0))
	slope = math.tan(math.radians(roll_deg)) * (w / 2.0)
	y0 = int(np.clip(horizon_y - slope, 0, h - 1))
	y1 = int(np.clip(horizon_y + slope, 0, h - 1))
	cv2.line(frame, (0, y0), (w - 1, y1), (0, 255, 255), 2, cv2.LINE_AA)


def crop_frame_to_fov(frame: np.ndarray, state: ViewerState, out_w: int = 480, out_h: int = 360) -> np.ndarray:
	if frame.size == 0:
		return frame
	in_h, in_w = frame.shape[:2]
	yaw_deg, pitch_deg, roll_deg = get_effective_orientation(state, state.frame_idx)
	map_x, map_y = build_remap(
		in_h=in_h,
		in_w=in_w,
		out_h=out_h,
		out_w=out_w,
		yaw_deg=yaw_deg,
		pitch_deg=pitch_deg,
		roll_deg=roll_deg,
		fov_deg=state.fov,
	)
	return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def make_raft_preview_pair(input_crop: np.ndarray, flow_img: np.ndarray) -> np.ndarray:
	left = input_crop.copy()
	right = flow_img.copy()
	if left.ndim == 2:
		left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
	if right.ndim == 2:
		right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
	if right.shape[:2] != left.shape[:2]:
		right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
	panel = np.hstack((left, right))
	cv2.putText(panel, "input crop", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(panel, "RAFT output", (left.shape[1] + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
	return panel


def draw_overlay(frame: np.ndarray, state: ViewerState, fps: float, total_frames: int) -> None:
	if not state.show_info:
		return

	label, out_w, out_h = ASPECT_PRESETS[state.preset_idx]
	play_state = "paused" if state.paused else "playing"
	level_status = "auto-level on" if state.use_auto_level else "auto-level off"
	yaw_status = yaw_mode_label(state.yaw_mode)
	line1 = f"{play_state} frame={state.frame_idx}/{max(total_frames - 1, 0)}  fps={fps:.2f}"
	line2 = (
		f"yaw={state.yaw:.1f} pitch={state.pitch:.1f} roll={state.roll:.1f} "
		f"fov={state.fov:.1f} preset={label} {out_w}x{out_h}"
	)
	line3 = f"{level_status} | {yaw_status} | telemetry={state.telemetry.source or 'none'}"
	time_sec = state.frame_idx / max(fps, 1e-6)
	grav_value = sample_stream_at_time(state.telemetry.grav, state.telemetry.grav_timestamps, time_sec)
	cori_value = sample_stream_at_time(state.telemetry.cori, state.telemetry.cori_timestamps, time_sec)
	shut_value = sample_stream_at_time(state.telemetry.shut, state.telemetry.shut_timestamps, time_sec)
	isoe_value = sample_stream_at_time(state.telemetry.isoe, state.telemetry.isoe_timestamps, time_sec)
	line4 = f"t={time_sec:.3f}s GRAV={format_sample(grav_value, 5)} CORI={format_sample(cori_value, 5)}"
	line5 = f"SHUT={format_sample(shut_value, 7)} ISOE={format_sample(isoe_value, 2)}"
	line6 = "drag=pan/tilt wheel=fov z/x=roll a=level y=yaw-mode r=aspect p=play/pause s=save i=info q/esc=quit"

	panel_lines = [line1, line2, line3, line4, line5, line6]
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.58
	thickness = 1
	line_gap = 8
	margin = 10
	baseline = 0

	max_w = 0
	total_h = margin
	line_sizes = []
	for text in panel_lines:
		(text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
		line_sizes.append((text_w, text_h, baseline))
		max_w = max(max_w, text_w)
		total_h += text_h + line_gap
	total_h += margin - line_gap

	panel_w = max_w + margin * 2
	panel_h = total_h
	cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), (90, 90, 90), thickness=-1)

	y = 8 + margin
	for idx, text in enumerate(panel_lines):
		_, text_h, _ = line_sizes[idx]
		y += text_h
		cv2.putText(frame, text, (8 + margin, y), font, font_scale, (235, 235, 235), thickness, cv2.LINE_AA)
		y += line_gap

	now = time.time()
	if state.status_text and now <= state.status_until:
		(text_w, text_h), _ = cv2.getTextSize(state.status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
		x0 = 8
		y0 = frame.shape[0] - (text_h + 20)
		cv2.rectangle(frame, (x0, y0), (x0 + text_w + 20, y0 + text_h + 14), (90, 90, 90), thickness=-1)
		cv2.putText(frame, state.status_text, (x0 + 10, y0 + text_h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 245, 120), 1, cv2.LINE_AA)


def choose_export_size(src_w: int, src_h: int, preset_idx: int) -> tuple[int, int]:
	_, preset_w, preset_h = ASPECT_PRESETS[preset_idx]
	aspect = preset_w / preset_h

	candidate_h = round_even(src_w / aspect)
	if candidate_h <= src_h and candidate_h > 0:
		return round_even(src_w), candidate_h

	candidate_w = round_even(src_h * aspect)
	return candidate_w, round_even(src_h)


def make_output_path(input_path: Path, user_output: str) -> Path:
	if user_output:
		return Path(user_output)
	return input_path.with_name(f"{input_path.stem}_dewarped.mp4")


def create_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
	candidates = [("avc1", "H.264 (avc1)"), ("H264", "H.264 (H264)"), ("mp4v", "MPEG-4 Part 2 (mp4v)")]
	for fourcc_name, label in candidates:
		writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), fps, size)
		if writer.isOpened():
			return writer, label
		writer.release()
	raise RuntimeError("Could not open VideoWriter with avc1/H264/mp4v")


def run_export(
	input_path: Path,
	output_path: Path,
	yaw: float,
	pitch: float,
	roll: float,
	fov: float,
	preset_idx: int,
	use_auto_level: bool,
	yaw_mode: str,
	telemetry: TelemetryData,
	status: ExportStatus,
	lock: threading.Lock,
) -> None:
	cap = cv2.VideoCapture(str(input_path))
	if not cap.isOpened():
		with lock:
			status.done = True
			status.success = False
			status.message = "Export failed: could not open input video"
			status.active = False
		return

	src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	out_w, out_h = choose_export_size(src_w, src_h, preset_idx)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	try:
		writer, codec_label = create_writer(output_path, fps, (out_w, out_h))
	except RuntimeError as err:
		cap.release()
		with lock:
			status.done = True
			status.success = False
			status.message = f"Export failed: {err}"
			status.active = False
		return

	print(f"Export started -> {output_path}")
	print(f"Codec: {codec_label} | Size: {out_w}x{out_h} | FPS: {fps:.3f}")

	processed = 0
	started = time.time()
	last_print = 0.0
	success = True
	message = ""

	while True:
		ok, frame = cap.read()
		if not ok:
			break
		effective_yaw, effective_pitch, effective_roll = get_effective_orientation(
			ViewerState(yaw=yaw, pitch=pitch, roll=roll, fov=fov, preset_idx=preset_idx, telemetry=telemetry, use_auto_level=use_auto_level, yaw_mode=yaw_mode),
			processed,
		)
		map_x, map_y = build_remap(
			in_h=src_h,
			in_w=src_w,
			out_h=out_h,
			out_w=out_w,
			yaw_deg=effective_yaw,
			pitch_deg=effective_pitch,
			roll_deg=effective_roll,
			fov_deg=fov,
		)
		dewarped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
		writer.write(dewarped)
		processed += 1

		now = time.time()
		if now - last_print >= 0.25:
			elapsed = max(now - started, 1e-6)
			fps_now = processed / elapsed
			if total > 0:
				pct = 100.0 * (processed / total)
				eta = (total - processed) / max(fps_now, 1e-6)
				print(
					f"\rExport progress: {processed}/{total} ({pct:5.1f}%) | {fps_now:5.1f} fps | ETA {eta:6.1f}s",
					end="",
					flush=True,
				)
			else:
				print(
					f"\rExport progress: {processed} frames | {fps_now:5.1f} fps",
					end="",
					flush=True,
				)
			last_print = now

		with lock:
			status.processed = processed
			status.total = total

	if processed == 0:
		success = False
		message = "Export failed: no frames were processed"

	cap.release()
	writer.release()

	elapsed = max(time.time() - started, 1e-6)
	print()
	if success:
		print(f"Export complete in {elapsed:.1f}s -> {output_path}")
		message = f"Saved {output_path.name}"
	else:
		print(message)

	with lock:
		status.done = True
		status.success = success
		status.message = message
		status.active = False
		status.processed = processed
		status.total = total


def start_export(state: ViewerState, input_path: Path, output_path: Path) -> None:
	with state.export_lock:
		if state.export_status.active:
			state.status_text = "Export already in progress"
			state.status_until = time.time() + 2.0
			return

		state.export_status = ExportStatus(
			active=True,
			done=False,
			success=False,
			message="",
			output_path=str(output_path),
			processed=0,
			total=0,
			started_at=time.time(),
		)

		thread = threading.Thread(
			target=run_export,
			args=(
				input_path,
				output_path,
				state.yaw,
				state.pitch,
				state.roll,
				state.fov,
				state.preset_idx,
				state.use_auto_level,
				state.yaw_mode,
				state.telemetry,
				state.export_status,
				state.export_lock,
			),
			daemon=True,
		)
		state.export_thread = thread
		thread.start()

	state.status_text = f"Export started: {output_path.name}"
	state.status_until = time.time() + 3.0


def run_raft_crop_preview(state: ViewerState, input_path: Path, raft_onnx_path: Path) -> None:
	cap = cv2.VideoCapture(str(input_path))
	if not cap.isOpened():
		with state.export_lock:
			state.raft_active = False
			state.raft_status = "RAFT failed: could not open input video"
			state.raft_input_crop = None
			state.raft_output = None
		return

	start_frame = max(0, min(state.frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	ok, prev_frame = cap.read()
	if not ok:
		cap.release()
		with state.export_lock:
			state.raft_active = False
			state.raft_status = "RAFT failed: no preview frames available"
			state.raft_input_crop = None
			state.raft_output = None
		return

	try:
		flow_estimator = Raft(str(raft_onnx_path))
	except Exception:
		cap.release()
		with state.export_lock:
			state.raft_active = False
			state.raft_status = "RAFT failed: could not load ONNX model"
			state.raft_input_crop = None
			state.raft_output = None
		return

	crop_prev = crop_frame_to_fov(prev_frame, state, out_w=480, out_h=360)
	frame_count = 0
	max_preview_frames = 12
	last_status = time.time()
	try:
		while frame_count < max_preview_frames:
			ok, frame = cap.read()
			if not ok:
				break
			frame_count += 1
			crop_curr = crop_frame_to_fov(frame, state, out_w=480, out_h=360)
			flow_map = flow_estimator(crop_prev, crop_curr)
			flow_img = flow_estimator.draw_flow()
			with state.export_lock:
				state.raft_input_crop = crop_curr.copy()
				state.raft_output = flow_img.copy()
				state.raft_status = f"RAFT {frame_count}/{max_preview_frames}"
			prev_frame = frame
			crop_prev = crop_curr
			if time.time() - last_status >= 0.1:
				last_status = time.time()
	finally:
		cap.release()
		with state.export_lock:
			state.raft_active = False
			state.raft_status = "RAFT crop ready"


def set_temp_status(state: ViewerState, text: str, seconds: float = 1.5) -> None:
	state.status_text = text
	state.status_until = time.time() + seconds


def mouse_callback(event: int, x: int, y: int, flags: int, userdata: ViewerState) -> None:
	state = userdata
	if event == cv2.EVENT_LBUTTONDOWN:
		state.dragging = True
		state.last_mouse_x = x
		state.last_mouse_y = y
	elif event == cv2.EVENT_LBUTTONUP:
		state.dragging = False
	elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
		dx = x - state.last_mouse_x
		dy = y - state.last_mouse_y
		state.last_mouse_x = x
		state.last_mouse_y = y

		state.yaw -= dx * 0.15
		state.pitch = clamp(state.pitch - dy * 0.15, -89.0, 89.0)
		state.dirty = True
		set_temp_status(state, f"Yaw {state.yaw:.1f}  Pitch {state.pitch:.1f}", 0.7)
	elif event == cv2.EVENT_MOUSEWHEEL:
		wheel_delta = 1 if flags > 0 else -1
		state.fov = clamp(state.fov - wheel_delta * 2.0, MIN_FOV, MAX_FOV)
		state.dirty = True
		set_temp_status(state, f"FOV {state.fov:.1f}", 0.7)


def start_raft_preview(state: ViewerState, input_path: Path, raft_onnx_path: Path) -> None:
	with state.export_lock:
		if state.raft_active:
			set_temp_status(state, "RAFT already running", 1.2)
			return
		state.raft_active = True
		state.raft_status = "Starting RAFT crop..."
		state.raft_input_crop = None
		state.raft_output = None
	state.raft_thread = threading.Thread(
		target=run_raft_crop_preview,
		args=(state, input_path, raft_onnx_path),
		daemon=True,
	)
	state.raft_thread.start()
	set_temp_status(state, "Running RAFT crop...", 1.5)


def process_key(key: int, state: ViewerState, input_path: Path, output_path: Path, raft_onnx_path: Path) -> bool:
	if key in (27, ord("q")):
		return False
	if key == ord("r"):
		start_raft_preview(state, input_path, raft_onnx_path)
	elif key == ord("t"):
		state.preset_idx = (state.preset_idx + 1) % len(ASPECT_PRESETS)
		_, out_w, out_h = ASPECT_PRESETS[state.preset_idx]
		cv2.resizeWindow(WINDOW_NAME, out_w, out_h)
		state.dirty = True
		set_temp_status(state, f"Preset {ASPECT_PRESETS[state.preset_idx][0]} {out_w}x{out_h}")
	elif key == ord("z"):
		state.roll -= 1.0
		state.dirty = True
		set_temp_status(state, f"Roll {state.roll:.1f}")
	elif key == ord("x"):
		state.roll += 1.0
		state.dirty = True
		set_temp_status(state, f"Roll {state.roll:.1f}")
	elif key == ord("a"):
		state.use_auto_level = not state.use_auto_level
		state.dirty = True
		set_temp_status(state, "Auto-level on" if state.use_auto_level else "Auto-level off")
	elif key == ord("y"):
		modes = available_yaw_modes(state.telemetry)
		current_idx = modes.index(state.yaw_mode) if state.yaw_mode in modes else 0
		state.yaw_mode = modes[(current_idx + 1) % len(modes)]
		state.dirty = True
		set_temp_status(state, f"Yaw: {state.yaw_mode}")
	elif key == ord("p"):
		state.paused = not state.paused
		state.dirty = True
		set_temp_status(state, "Paused" if state.paused else "Playing")
	elif key == ord("i"):
		state.show_info = not state.show_info
		state.dirty = True
		set_temp_status(state, "Info hidden" if not state.show_info else "Info shown")
	elif key == ord("s"):
		start_export(state, input_path, output_path)
	return True


def run() -> int:
	args = parse_args()
	input_path = Path(args.input_video)
	if not input_path.exists():
		print(f"Input not found: {input_path}")
		return 1

	state = ViewerState(
		yaw=args.yaw,
		pitch=clamp(args.pitch, -89.0, 89.0),
		roll=args.roll,
		fov=clamp(args.fov, MIN_FOV, MAX_FOV),
		preset_idx=max(0, min(len(ASPECT_PRESETS) - 1, args.preset)),
	)

	output_path = make_output_path(input_path, args.output)

	cap = cv2.VideoCapture(str(input_path))
	if not cap.isOpened():
		print(f"Failed to open input video: {input_path}")
		return 1

	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	state.telemetry = load_gpmf_telemetry(input_path, max(total_frames, 1))
	if state.telemetry.gps_heading_deg is None:
		print("No GPS heading found in GPMF telemetry; press 'r' to run RAFT on the current crop.")
	if state.telemetry.source != "none":
		print(f"Loaded GPMF telemetry ({state.telemetry.source})")
	else:
		print("No GPMF telemetry found; continuing with manual controls")

	if state.telemetry.gps_heading_deg is None and state.telemetry.flow_heading_deg is not None:
		state.yaw_mode = "flow"

	ok, frame = cap.read()
	if not ok:
		print("Failed to read first frame")
		cap.release()
		return 1

	state.frame_idx = 0
	state.paused = True
	state.dirty = True

	cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
	cv2.namedWindow(RAFT_PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
	_, out_w, out_h = ASPECT_PRESETS[state.preset_idx]
	cv2.resizeWindow(WINDOW_NAME, out_w, out_h)
	cv2.resizeWindow(RAFT_PREVIEW_WINDOW, 1000, 420)
	cv2.setMouseCallback(WINDOW_NAME, mouse_callback, state)

	print("Controls:")
	print("  drag mouse: pan/tilt")
	print("  mouse wheel: adjust FOV")
	print("  r: run RAFT on the current direction crop")
	print("  t: cycle preview aspect preset")
	print("  z/x: roll -/+ 1 degree")
	print("  a: toggle auto-level from CORI telemetry")
	print("  y: cycle yaw mode (manual / GPS / flow)")
	print("  p: play/pause")
	print("  i: toggle on-screen info panel")
	print("  s: save full video from frame 0 using current settings")
	print("  q or Esc: quit")

	should_run = True
	while should_run:
		if not state.paused:
			ok, next_frame = cap.read()
			if ok:
				frame = next_frame
				state.frame_idx += 1
				state.dirty = True
			else:
				state.paused = True
				set_temp_status(state, "Reached end of video", 2.0)

		with state.export_lock:
			if state.export_status.done:
				set_temp_status(state, state.export_status.message or "Export finished", 4.0)
				state.export_status.done = False

		if state.dirty or state.paused:
			display = render_frame(frame, state)
			draw_horizon_overlay(display, state)
			draw_overlay(display, state, fps, total_frames)

			if state.raft_input_crop is not None and state.raft_output is not None:
				cv2.imshow(RAFT_PREVIEW_WINDOW, make_raft_preview_pair(state.raft_input_crop, state.raft_output))

			with state.export_lock:
				if state.export_status.active and state.show_info:
					elapsed = max(time.time() - state.export_status.started_at, 1e-6)
					pcount = state.export_status.processed
					tcount = state.export_status.total
					est = ""
					if tcount > 0 and pcount > 0:
						fps_now = pcount / elapsed
						eta = (tcount - pcount) / max(fps_now, 1e-6)
						pct = 100.0 * pcount / tcount
						est = f" {pct:5.1f}% ETA {eta:6.1f}s"
					export_line = f"exporting {pcount}/{tcount if tcount > 0 else '?'}{est}"
					(text_w, text_h), _ = cv2.getTextSize(export_line, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
					x0 = 8
					y0 = 126
					cv2.rectangle(display, (x0, y0), (x0 + text_w + 20, y0 + text_h + 14), (90, 90, 90), thickness=-1)
					cv2.putText(display, export_line, (x0 + 10, y0 + text_h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (100, 220, 255), 1, cv2.LINE_AA)

			cv2.imshow(WINDOW_NAME, display)
			state.dirty = False

		key = cv2.waitKey(10) & 0xFF
		if key != 255:
			should_run = process_key(key, state, input_path, output_path, Path(args.raft_onnx))

	cap.release()
	cv2.destroyWindow(RAFT_PREVIEW_WINDOW)
	cv2.destroyAllWindows()
	if state.export_thread and state.export_thread.is_alive():
		print("Waiting for active export to finish...")
		state.export_thread.join()

	return 0


if __name__ == "__main__":
	raise SystemExit(run())
