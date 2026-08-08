# gopro360

Interactive OpenCV viewer for 360-degree equirectangular video with live dewarp controls and full-video export.

## Run with uv

Install dependencies:

```bash
uv sync
```

Launch viewer:

```bash
uv run gopro360 /path/to/input.mp4
```

Launch with options:

```bash
uv run gopro360 /path/to/input.mp4 \
  --output /path/to/output.mp4 \
  --yaw 0 --pitch 0 --roll 0 --fov 90 --preset 1
```

## Controls

- Left mouse drag: pan (yaw) and tilt (pitch)
- Mouse wheel: adjust FOV
- `r`: cycle window aspect presets
  - 1:1 -> 1080x1080
  - 16:9 -> 1920x1080
  - 9:16 -> 1080x1920
  - 4:3 -> 1440x1080
- `z` / `x`: roll -/+ 1 degree
- `p`: pause or resume playback
- `i`: toggle on-screen info panel
- `s`: export full video from frame 0 using current settings snapshot
- `q` or `Esc`: quit

## Notes

- Viewer starts windowed (not fullscreen).
- First frame is shown immediately in dewarped perspective.
- Export prints progress in terminal and overlays export status in the viewer.
- Export target is MP4; writer prefers H.264 and falls back to mp4v if needed.

## Legacy ffmpeg notes

```
ffmpeg -y -ss 00:00:00 -to 00:00:28.0750000 -i /home/jason/Videos/Wye/GS010014.360 \
  -progress pipe:1 -stats_period 1.0 \
  -filter_complex "[0:0]crop=624:1344:x=0:y=0,format=yuvj420p[LEFTFRAME_left_slice]; \
[0:0]crop=128:1344:x=624:y=0,format=yuvj420p,geq=lum='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))':cb='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))':cr='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))':a='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))',crop=64:1344:x=0:y=0,format=yuvj420p,scale=96:1344[LEFTFRAME_overlap_slice]; \
[0:0]crop=624:1344:x=752:y=0,format=yuvj420p[LEFTFRAME_right_slice]; \
[LEFTFRAME_left_slice][LEFTFRAME_overlap_slice]hstack[LEFTFRAME_left_and_overlap_joined]; \
[LEFTFRAME_left_and_overlap_joined][LEFTFRAME_right_slice]hstack[LEFTFRAME_completed]; \
[0:0]crop=1344:1344:1376:0[CENTERFRAME_completed]; \
[0:0]crop=624:1344:x=2720:y=0,format=yuvj420p[RIGHTFRAME_left_slice]; \
[0:0]crop=128:1344:x=3344:y=0,format=yuvj420p,geq=lum='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))':cb='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))':cr='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))':a='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))',crop=64:1344:x=0:y=0,format=yuvj420p,scale=96:1344[RIGHTFRAME_overlap_slice]; \
[0:0]crop=624:1344:x=3472:y=0,format=yuvj420p[RIGHTFRAME_right_slice]; \
[RIGHTFRAME_left_slice][RIGHTFRAME_overlap_slice]hstack[RIGHTFRAME_left_and_overlap_joined]; \
[RIGHTFRAME_left_and_overlap_joined][RIGHTFRAME_right_slice]hstack[RIGHTFRAME_completed]; \
[LEFTFRAME_completed][CENTERFRAME_completed]hstack[LEFT_CENTER_frames_joined]; \
[LEFT_CENTER_frames_joined][RIGHTFRAME_completed]hstack[LEFT_CENTER_RIGHT_completed]; \
[0:4]crop=624:1344:x=0:y=0,format=yuvj420p[BOTTOMFRAME_left_slice]; \
[0:4]crop=128:1344:x=624:y=0,format=yuvj420p,geq=lum='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))':cb='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))':cr='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))':a='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))',crop=64:1344:x=0:y=0,format=yuvj420p,scale=96:1344[BOTTOMFRAME_overlap_slice]; \
[0:4]crop=624:1344:x=752:y=0,format=yuvj420p[BOTTOMFRAME_right_slice]; \
[BOTTOMFRAME_left_slice][BOTTOMFRAME_overlap_slice]hstack[BOTTOMFRAME_left_and_overlap_joined]; \
[BOTTOMFRAME_left_and_overlap_joined][BOTTOMFRAME_right_slice]hstack[BOTTOMFRAME_completed]; \
[0:4]crop=1344:1344:1376:0[BACKFRAME_completed]; \
[0:4]crop=624:1344:x=2720:y=0,format=yuvj420p[TOPFRAME_left_slice]; \
[0:4]crop=128:1344:x=3344:y=0,format=yuvj420p,geq=lum='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))':cb='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))':cr='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))':a='if(between(X, 0, 64), (p((X+64),Y)*(((X+1))/65))+(p(X,Y)*((65-((X+1)))/65)), p(X,Y))',crop=64:1344:x=0:y=0,format=yuvj420p,scale=96:1344[TOPFRAME_overlap_slice]; \
[0:4]crop=624:1344:x=3472:y=0,format=yuvj420p[TOPFRAME_right_slice]; \
[TOPFRAME_left_slice][TOPFRAME_overlap_slice]hstack[TOPFRAME_left_and_overlap_joined]; \
[TOPFRAME_left_and_overlap_joined][TOPFRAME_right_slice]hstack[TOPFRAME_completed]; \
[BOTTOMFRAME_completed][BACKFRAME_completed]hstack[BOTTOM_BACK_frames_joined]; \
[BOTTOM_BACK_frames_joined][TOPFRAME_completed]hstack[BOTTOM_BACK_TOP_completed]; \
[LEFT_CENTER_RIGHT_completed][BOTTOM_BACK_TOP_completed]vstack[FULL_EAC_FRAME]; \
[FULL_EAC_FRAME]v360=eac:equirect:interp=cubic:roll=0:pitch=0:yaw=0,crop=4032:2388:x=0:y=0[OUTPUT_FRAME]" \
  -map 0:a:0 -map "[OUTPUT_FRAME]" \
  -c:a pcm_s16le -c:v prores -f mov \
  /home/jason/Videos/Wye/GS010014-00-00-00__00-00-28.mov



# 1. Scale video to 2:1 ratio with faststart header
ffmpeg -i GS010019-00-00-02__00-00-04Apple.mov -vf "scale=4032:2016" -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart -c:a aac -b:a 192k output_insta.mp4

```