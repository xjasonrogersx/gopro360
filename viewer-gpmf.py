from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "gopro360 viewer"
MIN_FOV = 30.0
MAX_FOV = 170.0
ASPECT_PRESETS = [
	("1:1", 1080, 1080),
	("16:9", 1920, 1080),
	("9:16", 1080, 1920),
	("4:3", 1440, 1080),
]


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
	return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
	return max(low, min(high, value))


def round_even(value: float) -> int:
	rounded = int(round(value))
	return rounded if rounded % 2 == 0 else rounded - 1


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


def get_cached_remap(state: ViewerState, in_h: int, in_w: int, out_h: int, out_w: int) -> tuple[np.ndarray, np.ndarray]:
	key = (
		in_h,
		in_w,
		out_h,
		out_w,
		round(state.yaw, 4),
		round(state.pitch, 4),
		round(state.roll, 4),
		round(state.fov, 4),
	)
	if state.map_key != key or state.map_x is None or state.map_y is None:
		state.map_x, state.map_y = build_remap(
			in_h=in_h,
			in_w=in_w,
			out_h=out_h,
			out_w=out_w,
			yaw_deg=state.yaw,
			pitch_deg=state.pitch,
			roll_deg=state.roll,
			fov_deg=state.fov,
		)
		state.map_key = key
	return state.map_x, state.map_y


def render_frame(frame: np.ndarray, state: ViewerState) -> np.ndarray:
	_, out_w, out_h = ASPECT_PRESETS[state.preset_idx]
	in_h, in_w = frame.shape[:2]
	map_x, map_y = get_cached_remap(state, in_h, in_w, out_h, out_w)
	return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def draw_overlay(frame: np.ndarray, state: ViewerState, fps: float, total_frames: int) -> None:
	if not state.show_info:
		return

	label, out_w, out_h = ASPECT_PRESETS[state.preset_idx]
	play_state = "paused" if state.paused else "playing"
	line1 = f"{play_state} frame={state.frame_idx}/{max(total_frames - 1, 0)}  fps={fps:.2f}"
	line2 = (
		f"yaw={state.yaw:.1f} pitch={state.pitch:.1f} roll={state.roll:.1f} "
		f"fov={state.fov:.1f} preset={label} {out_w}x{out_h}"
	)
	line3 = "drag=pan/tilt wheel=fov z/x=roll r=aspect p=play/pause s=save i=info q/esc=quit"

	panel_lines = [line1, line2, line3]
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

	map_x, map_y = build_remap(
		in_h=src_h,
		in_w=src_w,
		out_h=out_h,
		out_w=out_w,
		yaw_deg=yaw,
		pitch_deg=pitch,
		roll_deg=roll,
		fov_deg=fov,
	)

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
				state.export_status,
				state.export_lock,
			),
			daemon=True,
		)
		state.export_thread = thread
		thread.start()

	state.status_text = f"Export started: {output_path.name}"
	state.status_until = time.time() + 3.0


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


def process_key(key: int, state: ViewerState, input_path: Path, output_path: Path) -> bool:
	if key in (27, ord("q")):
		return False
	if key == ord("r"):
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

	ok, frame = cap.read()
	if not ok:
		print("Failed to read first frame")
		cap.release()
		return 1

	state.frame_idx = 0
	state.paused = True
	state.dirty = True

	cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
	_, out_w, out_h = ASPECT_PRESETS[state.preset_idx]
	cv2.resizeWindow(WINDOW_NAME, out_w, out_h)
	cv2.setMouseCallback(WINDOW_NAME, mouse_callback, state)

	print("Controls:")
	print("  drag mouse: pan/tilt")
	print("  mouse wheel: adjust FOV")
	print("  r: cycle preview aspect preset")
	print("  z/x: roll -/+ 1 degree")
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
			draw_overlay(display, state, fps, total_frames)

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
			should_run = process_key(key, state, input_path, output_path)

	cap.release()
	cv2.destroyAllWindows()
	if state.export_thread and state.export_thread.is_alive():
		print("Waiting for active export to finish...")
		state.export_thread.join()

	return 0


if __name__ == "__main__":
	raise SystemExit(run())
