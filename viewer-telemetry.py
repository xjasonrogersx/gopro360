from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import telemetry_parser
except Exception:  # pragma: no cover - depends on optional runtime package
    telemetry_parser = None

try:
    from gpmf.gps import extract_gps_blocks, parse_gps_block
    from gpmf.io import extract_gpmf_stream
except Exception:  # pragma: no cover - depends on optional runtime package
    extract_gpmf_stream = None
    extract_gps_blocks = None
    parse_gps_block = None


WINDOW_NAME = "gopro360 telemetry viewer"
MIN_FOV = 30.0
MAX_FOV = 170.0
ROLL_BIAS_TRIGGER_MIN = 60.0
ROLL_BIAS_TRIGGER_MAX = 120.0
HORIZON_COLOR = (0, 255, 255)
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
class TelemetrySample:
    t_ms: float
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_rate_dps: float | None = None


@dataclass
class CourseSample:
    t_ms: float
    course_deg: float
    speed_mps: float
    latitude: float | None = None
    longitude: float | None = None


@dataclass
class TelemetryFrameInfo:
    t_ms: float = 0.0
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_rate_dps: float | None = None
    course_deg: float | None = None
    speed_mps: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    yaw_source: str = "none"


@dataclass
class TelemetryData:
    imu_samples: list[TelemetrySample] = field(default_factory=list)
    course_samples: list[CourseSample] = field(default_factory=list)
    loaded: bool = False
    source: str = "none"
    status: str = ""
    roll_bias_deg: float = 0.0


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

    auto_level_enabled: bool = True
    auto_yaw_enabled: bool = True
    auto_roll: float = 0.0
    auto_pitch: float = 0.0
    auto_yaw: float = 0.0
    telemetry_offset_ms: float = 0.0
    current_video_t_ms: float = 0.0
    telemetry_data: TelemetryData = field(default_factory=TelemetryData)
    telemetry_frame: TelemetryFrameInfo = field(default_factory=TelemetryFrameInfo)
    last_auto_t_ms: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telemetry-aware 360 equirectangular OpenCV viewer")
    parser.add_argument("input_video", help="Path to equirectangular input video")
    parser.add_argument(
        "--output",
        default="",
        help="Output path for `s` export. Defaults to <input_stem>_dewarped.mp4",
    )
    parser.add_argument("--yaw", type=float, default=0.0, help="Initial manual yaw offset in degrees")
    parser.add_argument("--pitch", type=float, default=0.0, help="Initial manual pitch offset in degrees")
    parser.add_argument("--roll", type=float, default=0.0, help="Initial manual roll offset in degrees")
    parser.add_argument("--fov", type=float, default=90.0, help="Initial horizontal FOV in degrees")
    parser.add_argument(
        "--preset",
        type=int,
        default=1,
        help="Initial aspect preset index: 0=1:1, 1=16:9, 2=9:16, 3=4:3",
    )
    parser.add_argument(
        "--telemetry-offset-ms",
        type=float,
        default=0.0,
        help="Shift telemetry timing relative to video time in milliseconds",
    )
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle_deg(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


def blend_angle_deg(current: float, target: float, alpha: float) -> float:
    delta = wrap_angle_deg(target - current)
    return wrap_angle_deg(current + (delta * clamp(alpha, 0.0, 1.0)))


def round_even(value: float) -> int:
    rounded = int(round(value))
    return rounded if rounded % 2 == 0 else rounded - 1


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _get_first_float(obj: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in obj:
            out = _to_float(obj.get(key))
            if out is not None:
                return out
    return None


def _vector3_from_obj(obj: Any) -> tuple[float, float, float] | None:
    if isinstance(obj, dict):
        x = _get_first_float(obj, ["x", "X", "0"])
        y = _get_first_float(obj, ["y", "Y", "1"])
        z = _get_first_float(obj, ["z", "Z", "2"])
        if x is not None and y is not None and z is not None:
            return x, y, z
        return None
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        x = _to_float(obj[0])
        y = _to_float(obj[1])
        z = _to_float(obj[2])
        if x is not None and y is not None and z is not None:
            return x, y, z
    return None


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2r)
    x = (math.cos(lat1r) * math.sin(lat2r)) - (math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
    return wrap_angle_deg(math.degrees(math.atan2(y, x)))


def _estimate_roll_bias(samples: list[TelemetrySample]) -> float:
    roll_values = [s.roll_deg for s in samples if s.roll_deg is not None]
    if len(roll_values) < 10:
        return 0.0

    arr = np.array(roll_values[: min(len(roll_values), 500)], dtype=np.float32)
    median_roll = float(np.median(arr))
    abs_roll = abs(median_roll)

    if ROLL_BIAS_TRIGGER_MIN <= abs_roll <= ROLL_BIAS_TRIGGER_MAX:
        return -90.0 if median_roll > 0 else 90.0
    return 0.0


def _apply_roll_bias(samples: list[TelemetrySample], roll_bias_deg: float) -> None:
    if abs(roll_bias_deg) < 1e-6:
        return
    for s in samples:
        if s.roll_deg is not None:
            s.roll_deg = wrap_angle_deg(s.roll_deg + roll_bias_deg)


def _interp_scalar(samples_t: list[float], values: list[float], t_ms: float) -> float | None:
    if not samples_t:
        return None
    if t_ms <= samples_t[0]:
        return values[0]
    if t_ms >= samples_t[-1]:
        return values[-1]

    lo = 0
    hi = len(samples_t) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if samples_t[mid] <= t_ms:
            lo = mid
        else:
            hi = mid

    t0, t1 = samples_t[lo], samples_t[hi]
    if t1 <= t0:
        return values[lo]
    alpha = (t_ms - t0) / (t1 - t0)
    return values[lo] + ((values[hi] - values[lo]) * alpha)


def _interp_angle(samples_t: list[float], values_deg: list[float], t_ms: float) -> float | None:
    if not samples_t:
        return None
    if t_ms <= samples_t[0]:
        return wrap_angle_deg(values_deg[0])
    if t_ms >= samples_t[-1]:
        return wrap_angle_deg(values_deg[-1])

    lo = 0
    hi = len(samples_t) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if samples_t[mid] <= t_ms:
            lo = mid
        else:
            hi = mid

    t0, t1 = samples_t[lo], samples_t[hi]
    if t1 <= t0:
        return wrap_angle_deg(values_deg[lo])
    alpha = (t_ms - t0) / (t1 - t0)
    delta = wrap_angle_deg(values_deg[hi] - values_deg[lo])
    return wrap_angle_deg(values_deg[lo] + (delta * alpha))


def _build_imu_samples(raw_imu: list[dict[str, Any]]) -> list[TelemetrySample]:
    samples: list[TelemetrySample] = []
    filt_accel: tuple[float, float, float] | None = None
    alpha = 0.08

    for row in raw_imu:
        if not isinstance(row, dict):
            continue

        t_ms = _get_first_float(row, ["timestamp_ms", "timestampMs", "timestamp", "time_ms", "time"])
        if t_ms is None:
            continue

        # Some parsers report seconds in "timestamp".
        if t_ms < 10_000.0:
            t_ms *= 1000.0

        gyro = _vector3_from_obj(row.get("gyro") or row.get("gyroscope") or row.get("gyr"))
        accel = _vector3_from_obj(row.get("accl") or row.get("accel") or row.get("accelerometer") or row.get("acc"))

        roll_deg = None
        pitch_deg = None
        if accel is not None:
            if filt_accel is None:
                filt_accel = accel
            else:
                filt_accel = (
                    ((1.0 - alpha) * filt_accel[0]) + (alpha * accel[0]),
                    ((1.0 - alpha) * filt_accel[1]) + (alpha * accel[1]),
                    ((1.0 - alpha) * filt_accel[2]) + (alpha * accel[2]),
                )
            ax, ay, az = filt_accel
            roll_deg = math.degrees(math.atan2(ax, az if abs(az) > 1e-6 else 1e-6))
            pitch_deg = math.degrees(math.atan2(-ay, math.sqrt((ax * ax) + (az * az)) + 1e-6))

        yaw_rate_dps = None
        if gyro is not None:
            # Use Y axis as yaw-rate proxy for our camera-space convention.
            yaw_rate_dps = gyro[1]

        samples.append(TelemetrySample(t_ms=t_ms, roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_rate_dps=yaw_rate_dps))

    samples.sort(key=lambda s: s.t_ms)
    if not samples:
        return samples

    t0 = samples[0].t_ms
    for sample in samples:
        sample.t_ms -= t0
    return samples


def _build_course_samples_from_gpmf(video_path: Path) -> list[CourseSample]:
    if extract_gpmf_stream is None or extract_gps_blocks is None or parse_gps_block is None:
        return []

    stream = extract_gpmf_stream(str(video_path))
    blocks = [parse_gps_block(block) for block in extract_gps_blocks(stream)]
    if not blocks:
        return []

    points: list[tuple[datetime, float, float, float]] = []
    for block in blocks:
        base_time = datetime.strptime(block.timestamp, "%Y-%m-%d %H:%M:%S.%f")
        for i in range(block.npoints):
            t = base_time + timedelta(seconds=(i / 18.0))
            points.append((t, float(block.latitude[i]), float(block.longitude[i]), float(block.speed_2d[i])))

    if len(points) < 2:
        return []

    points.sort(key=lambda p: p[0])
    t0 = points[0][0]
    out: list[CourseSample] = []

    for i in range(1, len(points)):
        prev = points[i - 1]
        curr = points[i]
        course = _bearing_deg(prev[1], prev[2], curr[1], curr[2])
        t_ms = (curr[0] - t0).total_seconds() * 1000.0
        out.append(
            CourseSample(
                t_ms=t_ms,
                course_deg=course,
                speed_mps=max(0.0, curr[3]),
                latitude=curr[1],
                longitude=curr[2],
            )
        )

    return out


def load_telemetry(video_path: Path) -> TelemetryData:
    data = TelemetryData()
    sources: list[str] = []
    status_bits: list[str] = []

    if telemetry_parser is None:
        status_bits.append("telemetry-parser not available")
    else:
        try:
            parser = telemetry_parser.Parser(str(video_path))
            raw_imu = parser.normalized_imu()
            data.imu_samples = _build_imu_samples(raw_imu if isinstance(raw_imu, list) else [])
            if data.imu_samples:
                data.roll_bias_deg = _estimate_roll_bias(data.imu_samples)
                _apply_roll_bias(data.imu_samples, data.roll_bias_deg)
                sources.append("telemetry-parser")
                status_bits.append(f"imu={len(data.imu_samples)}")
                if abs(data.roll_bias_deg) > 0.0:
                    status_bits.append(f"roll-bias={data.roll_bias_deg:+.1f}")
            else:
                status_bits.append("no IMU from telemetry-parser")
        except Exception as exc:
            status_bits.append(f"telemetry-parser failed: {exc}")

    try:
        data.course_samples = _build_course_samples_from_gpmf(video_path)
        if data.course_samples:
            sources.append("gpmf-gps")
            status_bits.append(f"course={len(data.course_samples)}")
        else:
            status_bits.append("no GPS course from gpmf")
    except Exception as exc:
        status_bits.append(f"gpmf failed: {exc}")

    data.loaded = bool(data.imu_samples or data.course_samples)
    data.source = "+".join(sources) if sources else "none"
    data.status = "; ".join(status_bits)
    return data


def interpolate_imu_sample(samples: list[TelemetrySample], t_ms: float) -> TelemetrySample | None:
    if not samples:
        return None

    ts = [s.t_ms for s in samples]

    roll_values = [s.roll_deg for s in samples if s.roll_deg is not None]
    pitch_values = [s.pitch_deg for s in samples if s.pitch_deg is not None]
    yaw_values = [s.yaw_rate_dps for s in samples if s.yaw_rate_dps is not None]

    roll_ts = [s.t_ms for s in samples if s.roll_deg is not None]
    pitch_ts = [s.t_ms for s in samples if s.pitch_deg is not None]
    yaw_ts = [s.t_ms for s in samples if s.yaw_rate_dps is not None]

    roll = _interp_scalar(roll_ts, roll_values, t_ms) if roll_ts else None
    pitch = _interp_scalar(pitch_ts, pitch_values, t_ms) if pitch_ts else None
    yaw_rate = _interp_scalar(yaw_ts, yaw_values, t_ms) if yaw_ts else None

    if roll is None and pitch is None and yaw_rate is None:
        return None
    return TelemetrySample(t_ms=t_ms, roll_deg=roll, pitch_deg=pitch, yaw_rate_dps=yaw_rate)


def interpolate_course(course_samples: list[CourseSample], t_ms: float) -> tuple[float, float, float | None, float | None] | None:
    if not course_samples:
        return None

    ts = [s.t_ms for s in course_samples]
    course_vals = [s.course_deg for s in course_samples]
    speed_vals = [s.speed_mps for s in course_samples]

    course = _interp_angle(ts, course_vals, t_ms)
    speed = _interp_scalar(ts, speed_vals, t_ms)
    lat_vals = [s.latitude for s in course_samples if s.latitude is not None]
    lat_ts = [s.t_ms for s in course_samples if s.latitude is not None]
    lon_vals = [s.longitude for s in course_samples if s.longitude is not None]
    lon_ts = [s.t_ms for s in course_samples if s.longitude is not None]

    lat = _interp_scalar(lat_ts, [float(v) for v in lat_vals], t_ms) if lat_ts else None
    lon = _interp_scalar(lon_ts, [float(v) for v in lon_vals], t_ms) if lon_ts else None

    if course is None or speed is None:
        return None
    return course, speed, lat, lon


def telemetry_horizon_line(
    frame_w: int,
    frame_h: int,
    h_fov_deg: float,
    telemetry_roll_deg: float | None,
    telemetry_pitch_deg: float | None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    cx = frame_w * 0.5
    cy = frame_h * 0.5

    roll = telemetry_roll_deg or 0.0
    pitch = telemetry_pitch_deg or 0.0

    # Convert horizontal FOV to vertical FOV for pitch->pixel shift.
    h_fov_rad = math.radians(clamp(h_fov_deg, 1.0, 179.0))
    v_fov_rad = 2.0 * math.atan(math.tan(h_fov_rad * 0.5) * (frame_h / max(frame_w, 1)))
    v_fov_deg = max(1e-6, math.degrees(v_fov_rad))

    # Positive camera pitch now moves the horizon down in the image.
    # The previous sign was inverted relative to the drag/view convention.
    y_center = cy - ((pitch / v_fov_deg) * frame_h)
    slope = math.tan(math.radians(-roll))

    x0 = 0.0
    y0 = y_center + (slope * (x0 - cx))
    x1 = float(frame_w - 1)
    y1 = y_center + (slope * (x1 - cx))

    y0 = float(clamp(y0, -frame_h * 2.0, frame_h * 3.0))
    y1 = float(clamp(y1, -frame_h * 2.0, frame_h * 3.0))
    return (int(x0), int(round(y0))), (int(x1), int(round(y1)))


def get_effective_angles(state: ViewerState) -> tuple[float, float, float]:
    yaw = state.yaw + (state.auto_yaw if state.auto_yaw_enabled else 0.0)
    pitch = state.pitch + (state.auto_pitch if state.auto_level_enabled else 0.0)
    roll = state.roll + (state.auto_roll if state.auto_level_enabled else 0.0)
    return wrap_angle_deg(yaw), clamp(pitch, -89.0, 89.0), wrap_angle_deg(roll)


def compute_auto_terms_for_time(
    telemetry_data: TelemetryData,
    t_ms: float,
    dt_ms: float,
    auto_roll: float,
    auto_pitch: float,
    auto_yaw: float,
    auto_level_enabled: bool,
    auto_yaw_enabled: bool,
) -> tuple[float, float, float]:
    if not telemetry_data.loaded:
        return auto_roll, auto_pitch, auto_yaw

    imu = interpolate_imu_sample(telemetry_data.imu_samples, t_ms)
    course = interpolate_course(telemetry_data.course_samples, t_ms)

    if auto_level_enabled and imu is not None:
        if imu.roll_deg is not None:
            auto_roll = ((auto_roll * 0.85) + ((-imu.roll_deg) * 0.15))
        if imu.pitch_deg is not None:
            auto_pitch = ((auto_pitch * 0.85) + ((-imu.pitch_deg) * 0.15))

    if auto_yaw_enabled:
        if course is not None and course[1] >= 0.8:
            auto_yaw = blend_angle_deg(auto_yaw, course[0], 0.12)
        elif imu is not None and imu.yaw_rate_dps is not None:
            auto_yaw = wrap_angle_deg(auto_yaw + (imu.yaw_rate_dps * (dt_ms / 1000.0)))

    return auto_roll, auto_pitch, auto_yaw


def get_video_time_ms(cap: cv2.VideoCapture, fallback_frame_idx: int, fps: float) -> float:
    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_msec > 0.0:
        return pos_msec
    return (fallback_frame_idx / max(fps, 1e-6)) * 1000.0


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

    rays = np.stack([xx, -yy, np.full_like(xx, focal)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    r_yaw = np.array(
        [[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]],
        dtype=np.float32,
    )
    r_pitch = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(pitch), -math.sin(pitch)], [0.0, math.sin(pitch), math.cos(pitch)]],
        dtype=np.float32,
    )
    r_roll = np.array(
        [[math.cos(roll), -math.sin(roll), 0.0], [math.sin(roll), math.cos(roll), 0.0], [0.0, 0.0, 1.0]],
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
    eff_yaw, eff_pitch, eff_roll = get_effective_angles(state)
    key = (
        in_h,
        in_w,
        out_h,
        out_w,
        round(eff_yaw, 4),
        round(eff_pitch, 4),
        round(eff_roll, 4),
        round(state.fov, 4),
    )
    if state.map_key != key or state.map_x is None or state.map_y is None:
        state.map_x, state.map_y = build_remap(
            in_h=in_h,
            in_w=in_w,
            out_h=out_h,
            out_w=out_w,
            yaw_deg=eff_yaw,
            pitch_deg=eff_pitch,
            roll_deg=eff_roll,
            fov_deg=state.fov,
        )
        state.map_key = key
    return state.map_x, state.map_y


def render_frame(frame: np.ndarray, state: ViewerState) -> np.ndarray:
    _, out_w, out_h = ASPECT_PRESETS[state.preset_idx]
    in_h, in_w = frame.shape[:2]
    map_x, map_y = get_cached_remap(state, in_h, in_w, out_h, out_w)
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def update_auto_orientation(state: ViewerState, fps: float, video_t_ms: float | None = None) -> None:
    if not state.telemetry_data.loaded:
        return

    if video_t_ms is None:
        video_t_ms = (state.frame_idx / max(fps, 1e-6)) * 1000.0

    t_ms = max(0.0, video_t_ms + state.telemetry_offset_ms)
    dt_ms = 0.0 if state.last_auto_t_ms is None else max(0.0, t_ms - state.last_auto_t_ms)

    imu = interpolate_imu_sample(state.telemetry_data.imu_samples, t_ms)
    course = interpolate_course(state.telemetry_data.course_samples, t_ms)

    frame_info = TelemetryFrameInfo(t_ms=t_ms)
    yaw_source = "none"

    if imu is not None:
        frame_info.roll_deg = imu.roll_deg
        frame_info.pitch_deg = imu.pitch_deg
        frame_info.yaw_rate_dps = imu.yaw_rate_dps

        if state.auto_level_enabled:
            if imu.roll_deg is not None:
                state.auto_roll = ((state.auto_roll * 0.85) + ((-imu.roll_deg) * 0.15))
            if imu.pitch_deg is not None:
                state.auto_pitch = ((state.auto_pitch * 0.85) + ((-imu.pitch_deg) * 0.15))

    if state.auto_yaw_enabled:
        if course is not None and course[1] >= 0.8:
            frame_info.course_deg = course[0]
            frame_info.speed_mps = course[1]
            frame_info.latitude = course[2]
            frame_info.longitude = course[3]
            state.auto_yaw = blend_angle_deg(state.auto_yaw, course[0], 0.12)
            yaw_source = "gps"
        elif imu is not None and imu.yaw_rate_dps is not None:
            state.auto_yaw = wrap_angle_deg(state.auto_yaw + (imu.yaw_rate_dps * (dt_ms / 1000.0)))
            yaw_source = "gyro"

    state.last_auto_t_ms = t_ms
    frame_info.yaw_source = yaw_source
    if course is not None:
        frame_info.course_deg = course[0]
        frame_info.speed_mps = course[1]
        frame_info.latitude = course[2]
        frame_info.longitude = course[3]
    state.telemetry_frame = frame_info


def draw_overlay(frame: np.ndarray, state: ViewerState, fps: float, total_frames: int) -> None:
    if not state.show_info:
        return

    label, out_w, out_h = ASPECT_PRESETS[state.preset_idx]
    eff_yaw, eff_pitch, eff_roll = get_effective_angles(state)
    play_state = "paused" if state.paused else "playing"

    line1 = f"{play_state} frame={state.frame_idx}/{max(total_frames - 1, 0)} fps={fps:.2f}"
    line2 = (
        f"view yaw={eff_yaw:.1f} pitch={eff_pitch:.1f} roll={eff_roll:.1f} fov={state.fov:.1f} "
        f"preset={label} {out_w}x{out_h}"
    )
    line3 = (
        f"manual y/p/r={state.yaw:.1f}/{state.pitch:.1f}/{state.roll:.1f} "
        f"auto y/p/r={state.auto_yaw:.1f}/{state.auto_pitch:.1f}/{state.auto_roll:.1f}"
    )

    if state.telemetry_data.loaded:
        tf = state.telemetry_frame
        line4 = (
            f"telemetry={state.telemetry_data.source} auto-level={'on' if state.auto_level_enabled else 'off'} "
            f"auto-yaw={'on' if state.auto_yaw_enabled else 'off'} yaw-src={tf.yaw_source}"
        )
        course_txt = "n/a" if tf.course_deg is None else f"{tf.course_deg:.1f}"
        speed_txt = "n/a" if tf.speed_mps is None else f"{tf.speed_mps:.2f}"
        lat_txt = "n/a" if tf.latitude is None else f"{tf.latitude:.6f}"
        lon_txt = "n/a" if tf.longitude is None else f"{tf.longitude:.6f}"
        line5 = (
            f"t={tf.t_ms/1000.0:.2f}s roll={tf.roll_deg if tf.roll_deg is not None else float('nan'):.2f} "
            f"pitch={tf.pitch_deg if tf.pitch_deg is not None else float('nan'):.2f} "
            f"course={course_txt} speed={speed_txt} rbias={state.telemetry_data.roll_bias_deg:+.1f}"
        )
        line6 = f"gps lat={lat_txt} lon={lon_txt} offset={state.telemetry_offset_ms:+.0f}ms"
    else:
        line4 = f"telemetry unavailable: {state.telemetry_data.status or 'no data'}"
        line5 = "auto-level and auto-yaw will not affect playback without telemetry data"
        line6 = f"gps lat=n/a lon=n/a offset={state.telemetry_offset_ms:+.0f}ms"

    line7 = "drag=pan/tilt wheel=fov z/x=roll r=aspect a=auto-level y=auto-yaw [/] sync p=play s=save i=info q/esc=quit"

    panel_lines = [line1, line2, line3, line4, line5, line6, line7]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.54
    thickness = 1
    line_gap = 7
    margin = 10

    max_w = 0
    total_h = margin
    sizes = []
    for text in panel_lines:
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        sizes.append((text_w, text_h))
        max_w = max(max_w, text_w)
        total_h += text_h + line_gap
    total_h += margin - line_gap

    panel_w = max_w + margin * 2
    panel_h = total_h
    cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), (90, 90, 90), thickness=-1)

    y = 8 + margin
    for idx, text in enumerate(panel_lines):
        _, text_h = sizes[idx]
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


def draw_horizon_line(frame: np.ndarray, state: ViewerState) -> None:
    if not state.telemetry_data.loaded:
        return

    # Draw the telemetry horizon as it appears in the current view.
    # Raw telemetry tilt is combined with the current effective camera tilt,
    # so dragging pitch/roll (or auto-level) updates the overlay line.
    _, eff_pitch, eff_roll = get_effective_angles(state)
    telemetry_roll = state.telemetry_frame.roll_deg or 0.0
    telemetry_pitch = state.telemetry_frame.pitch_deg or 0.0

    p0, p1 = telemetry_horizon_line(
        frame_w=frame.shape[1],
        frame_h=frame.shape[0],
        h_fov_deg=state.fov,
        telemetry_roll_deg=-(telemetry_roll + eff_roll),
        telemetry_pitch_deg=telemetry_pitch + eff_pitch,
    )
    cv2.line(frame, p0, p1, HORIZON_COLOR, 2, cv2.LINE_AA)


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
    manual_yaw: float,
    manual_pitch: float,
    manual_roll: float,
    fov: float,
    preset_idx: int,
    telemetry_data: TelemetryData,
    auto_level_enabled: bool,
    auto_yaw_enabled: bool,
    auto_roll_seed: float,
    auto_pitch_seed: float,
    auto_yaw_seed: float,
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

    auto_roll = auto_roll_seed
    auto_pitch = auto_pitch_seed
    auto_yaw = auto_yaw_seed
    last_t_ms: float | None = None

    print(f"Export started -> {output_path}")
    print(
        f"Codec: {codec_label} | Size: {out_w}x{out_h} | FPS: {fps:.3f} | "
        f"auto-level={'on' if auto_level_enabled else 'off'} auto-yaw={'on' if auto_yaw_enabled else 'off'}"
    )

    processed = 0
    started = time.time()
    last_print = 0.0
    success = True
    message = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_ms = (processed / max(fps, 1e-6)) * 1000.0
        dt_ms = 0.0 if last_t_ms is None else max(0.0, t_ms - last_t_ms)
        auto_roll, auto_pitch, auto_yaw = compute_auto_terms_for_time(
            telemetry_data=telemetry_data,
            t_ms=t_ms,
            dt_ms=dt_ms,
            auto_roll=auto_roll,
            auto_pitch=auto_pitch,
            auto_yaw=auto_yaw,
            auto_level_enabled=auto_level_enabled,
            auto_yaw_enabled=auto_yaw_enabled,
        )
        last_t_ms = t_ms

        yaw = wrap_angle_deg(manual_yaw + (auto_yaw if auto_yaw_enabled else 0.0))
        pitch = clamp(manual_pitch + (auto_pitch if auto_level_enabled else 0.0), -89.0, 89.0)
        roll = wrap_angle_deg(manual_roll + (auto_roll if auto_level_enabled else 0.0))

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
                print(f"\rExport progress: {processed} frames | {fps_now:5.1f} fps", end="", flush=True)
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
                state.telemetry_data,
                state.auto_level_enabled,
                state.auto_yaw_enabled,
                state.auto_roll,
                state.auto_pitch,
                state.auto_yaw,
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
        set_temp_status(state, f"Manual yaw {state.yaw:.1f}  pitch {state.pitch:.1f}", 0.7)
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
        set_temp_status(state, f"Manual roll {state.roll:.1f}")
    elif key == ord("x"):
        state.roll += 1.0
        state.dirty = True
        set_temp_status(state, f"Manual roll {state.roll:.1f}")
    elif key == ord("p"):
        state.paused = not state.paused
        state.dirty = True
        set_temp_status(state, "Paused" if state.paused else "Playing")
    elif key == ord("i"):
        state.show_info = not state.show_info
        state.dirty = True
        set_temp_status(state, "Info hidden" if not state.show_info else "Info shown")
    elif key == ord("a"):
        state.auto_level_enabled = not state.auto_level_enabled
        state.dirty = True
        set_temp_status(state, "Auto-level ON" if state.auto_level_enabled else "Auto-level OFF")
    elif key == ord("y"):
        state.auto_yaw_enabled = not state.auto_yaw_enabled
        state.dirty = True
        set_temp_status(state, "Auto-yaw ON" if state.auto_yaw_enabled else "Auto-yaw OFF")
    elif key == ord("["):
        state.telemetry_offset_ms -= 50.0
        state.dirty = True
        set_temp_status(state, f"Telemetry offset {state.telemetry_offset_ms:+.0f} ms")
    elif key == ord("]"):
        state.telemetry_offset_ms += 50.0
        state.dirty = True
        set_temp_status(state, f"Telemetry offset {state.telemetry_offset_ms:+.0f} ms")
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
        telemetry_offset_ms=args.telemetry_offset_ms,
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

    state.telemetry_data = load_telemetry(input_path)
    if state.telemetry_data.loaded:
        set_temp_status(state, f"Telemetry loaded: {state.telemetry_data.source}", 2.5)
        print(f"Telemetry: {state.telemetry_data.status}")
    else:
        set_temp_status(state, "Telemetry unavailable - manual controls only", 3.0)
        print(f"Telemetry unavailable: {state.telemetry_data.status}")

    state.frame_idx = 0
    state.paused = True
    state.dirty = True

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    _, out_w, out_h = ASPECT_PRESETS[state.preset_idx]
    cv2.resizeWindow(WINDOW_NAME, out_w, out_h)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, state)

    print("Controls:")
    print("  drag mouse: manual pan/tilt offsets")
    print("  mouse wheel: adjust FOV")
    print("  r: cycle preview aspect preset")
    print("  z/x: manual roll -/+ 1 degree")
    print("  a: toggle auto-level (default ON)")
    print("  y: toggle auto-yaw course vectoring (default ON)")
    print("  [ / ]: shift telemetry timing backward/forward 50 ms")
    print("  p: play/pause")
    print("  i: toggle on-screen info panel")
    print("  s: save full video from frame 0 using current effective view")
    print("  q or Esc: quit")

    should_run = True
    while should_run:
        if not state.paused:
            ok, next_frame = cap.read()
            if ok:
                frame = next_frame
                state.frame_idx += 1
                state.current_video_t_ms = get_video_time_ms(cap, state.frame_idx, fps)
                update_auto_orientation(state, fps, state.current_video_t_ms)
                state.dirty = True
            else:
                state.paused = True
                set_temp_status(state, "Reached end of video", 2.0)

        with state.export_lock:
            if state.export_status.done:
                set_temp_status(state, state.export_status.message or "Export finished", 4.0)
                state.export_status.done = False

        if state.paused and state.telemetry_data.loaded:
            # Keep telemetry values aligned with currently displayed frame while paused.
            update_auto_orientation(state, fps, state.current_video_t_ms)

        if state.dirty or state.paused:
            display = render_frame(frame, state)
            draw_horizon_line(display, state)
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
                    y0 = 164
                    cv2.rectangle(display, (x0, y0), (x0 + text_w + 20, y0 + text_h + 14), (90, 90, 90), thickness=-1)
                    cv2.putText(
                        display,
                        export_line,
                        (x0 + 10, y0 + text_h + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (100, 220, 255),
                        1,
                        cv2.LINE_AA,
                    )

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
