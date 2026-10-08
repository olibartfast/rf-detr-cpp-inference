#!/usr/bin/env bash
# check_display.sh — verify `--display` on a headless machine (Colab, a rented box, CI-like hosts).
#
# --display degrades silently: if SDL cannot open a window it prints "preview
# disabled" and the run carries on, so a plain headless run "passes" while showing
# nothing. This runs the app on a virtual X server and checks what a person at a
# screen would: a window of the video's size appears, it holds a real frame (not a
# blank surface), and pressing `q` ends the run early and cleanly.
#
#   APP=build-gpu/inference_app MODEL=m.onnx VIDEO=long.mp4 ./scripts/check_display.sh
#   ... ./scripts/check_display.sh --gpu-preprocess --gpu-postprocess --segmentation
#
# Extra arguments are passed to the app. Needs Xvfb, xdotool and ffmpeg (x11grab);
# python3 with Pillow and numpy for the pixel check. Exit status 0 only if every
# check passed. Rendering goes through Xvfb's software path: this proves the
# display code, not GPU-accelerated presentation.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
APP="${APP:-$REPO/build-gpu/inference_app}"
MODEL="${MODEL:?set MODEL to a .onnx or .engine}"
VIDEO="${VIDEO:?set VIDEO to a video file}"
LABELS="${LABELS:-$REPO/data/coco-labels-91.txt}"
OUT="${OUT:-$PWD/display-check}"
# Engine build from an .onnx happens before the window opens; allow for it.
WINDOW_TIMEOUT="${WINDOW_TIMEOUT:-900}"
TITLE="RF-DETR Inference" # src/video_pipeline.cpp

mkdir -p "$OUT"
# The app runs from $OUT (it writes output_video.mp4 to its cwd), so every
# path it is given must survive that cd.
APP="$(realpath "$APP")" MODEL="$(realpath "$MODEL")"
VIDEO="$(realpath "$VIDEO")" LABELS="$(realpath "$LABELS")"
LOG="$OUT/app.log"
status=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; status=1; }

for tool in Xvfb xdotool ffmpeg ffprobe python3; do
    command -v "$tool" >/dev/null || { echo "UNRUN display check — $tool not installed"; exit 2; }
done

IFS=x read -r vw vh < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
    -of csv=p=0:s=x "$VIDEO")
[[ "$vw" =~ ^[0-9]+$ && "$vh" =~ ^[0-9]+$ ]] || { echo "UNRUN display check — cannot read ${VIDEO}'s size"; exit 2; }
frames="$(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets \
    -of csv=p=0 "$VIDEO")"

# A free display number, so a second run does not collide with the first.
dnum=99
while [[ -e "/tmp/.X11-unix/X${dnum}" ]]; do dnum=$((dnum + 1)); done
Xvfb ":${dnum}" -screen 0 "$((vw + 64))x$((vh + 64))x24" -nolisten tcp > "$OUT/xvfb.log" 2>&1 &
xvfb_pid=$!
app_pid=""
# shellcheck disable=SC2329 # invoked by the EXIT trap
cleanup() {
    [[ -n "$app_pid" ]] && kill "$app_pid" 2>/dev/null
    kill "$xvfb_pid" 2>/dev/null
}
trap cleanup EXIT
export DISPLAY=":${dnum}" SDL_VIDEODRIVER=x11
sleep 1

(cd "$OUT" && exec "$APP" "$MODEL" "$VIDEO" "$LABELS" --display "$@") > "$LOG" 2>&1 &
app_pid=$!

wid=""
for ((i = 0; i < WINDOW_TIMEOUT; i++)); do
    wid="$(xdotool search --name "$TITLE" 2>/dev/null | head -1)"
    [[ -n "$wid" ]] && break
    kill -0 "$app_pid" 2>/dev/null || break
    sleep 1
done

if grep -q "preview disabled" "$LOG"; then
    fail "app disabled the preview: $(grep -m1 'preview disabled' "$LOG")"
fi
if [[ -z "$wid" ]]; then
    fail "no '${TITLE}' window within ${WINDOW_TIMEOUT}s — see $LOG"
    exit 1
fi
pass "window opened (id $wid)"

# Let a few frames through, then grab the window.
sleep 5
eval "$(xdotool getwindowgeometry --shell "$wid")" # sets X Y WIDTH HEIGHT
if [[ "$WIDTH" -eq "$vw" && "$HEIGHT" -eq "$vh" ]]; then
    pass "window is ${WIDTH}x${HEIGHT}, the video's size"
else
    fail "window is ${WIDTH}x${HEIGHT}, video is ${vw}x${vh}"
fi
ffmpeg -loglevel error -y -f x11grab -video_size "${WIDTH}x${HEIGHT}" -i "${DISPLAY}+${X},${Y}" \
    -frames:v 1 "$OUT/window.png"

# A blank or single-colour surface means the texture never reached the screen.
if python3 -I - "$OUT/window.png" <<'EOF'
import sys
import numpy as np
from PIL import Image
px = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float32)
std, distinct = float(px.std()), len(np.unique(px.reshape(-1, 3), axis=0))
print(f"window.png: std {std:.1f}, {distinct} distinct colours")
sys.exit(0 if std > 10 and distinct > 256 else 1)
EOF
then
    pass "window shows image content (window.png)"
else
    fail "window is blank or flat — see window.png"
fi

# `q` must end the run early and cleanly (src/display.cpp handles q and Escape).
xdotool key --window "$wid" q
for ((i = 0; i < 60; i++)); do
    kill -0 "$app_pid" 2>/dev/null || break
    sleep 1
done
if kill -0 "$app_pid" 2>/dev/null; then
    fail "app still running 60s after 'q'"
else
    wait "$app_pid"
    rc=$?
    app_pid=""
    done_frames="$(sed -n 's/^Processed \([0-9]*\) frames.*/\1/p' "$LOG")"
    if [[ "$rc" -eq 0 && -n "$done_frames" && "$done_frames" -lt "$frames" ]]; then
        pass "'q' stopped the run cleanly after ${done_frames}/${frames} frames"
    else
        fail "after 'q': exit ${rc}, processed '${done_frames}' of ${frames} — see $LOG"
    fi
fi

exit "$status"
