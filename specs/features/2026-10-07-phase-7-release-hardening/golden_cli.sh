#!/usr/bin/env bash
# Phase 7 behaviour-preservation check. Runs the CLI over a fixed workload and writes
# stdout, stderr, exit codes and output images to OUT_DIR, then hashes them.
#   golden_cli.sh <inference_app> <out_dir>
# Compare two runs with: diff <baseline>/SHA256SUMS <candidate>/SHA256SUMS
# Paths printed by the app are normalised, so baseline and candidate dirs may differ.
set -u
APP=$1
OUT=$2
# Data (gitignored models) lives in the main checkout, so resolve it even from a worktree.
REPO=${REPO:-$(dirname "$(git -C "$(dirname "$0")" rev-parse --path-format=absolute --git-common-dir)")}
CLIP=${CLIP:-/tmp/rfdetr-phase7-clip.mp4}
mkdir -p "$OUT"
cd "$REPO" || exit 2
L=data/coco-labels-91.txt
[ -f "$CLIP" ] || ffmpeg -loglevel error -y -loop 1 -i data/dog.jpg -t 2 -r 15 -pix_fmt yuv420p -vf scale=640:-2 "$CLIP"

run() { # name, args...
    local name=$1; shift
    "$APP" "$@" >"$OUT/$name.out" 2>"$OUT/$name.err"
    echo $? >"$OUT/$name.rc"
    sed -i -e "s#$OUT#<OUT>#g" -e "s#@ 0x[0-9a-f]*#@ <PTR>#g" "$OUT/$name.out" "$OUT/$name.err"
}
run det   data/models/rfdetr-nano-1101.onnx data/dog.jpg $L --output "$OUT/det.jpg"
run seg   data/models/rfdetr-seg-nano-576.onnx data/dog.jpg $L --segmentation --output "$OUT/seg.jpg"
run kp    data/models/rfdetr-keypoint-preview.onnx data/dog.jpg $L --keypoint --output "$OUT/kp.jpg"
run kpcnt data/models/rfdetr-keypoint-preview.onnx data/dog.jpg $L --keypoint --keypoint-counts 0,17 --output "$OUT/kpcnt.jpg"
run flags data/models/rfdetr-nano-1101.onnx data/dog.jpg $L --threshold 0.3 --max-detections 5 --background-class-id none --output "$OUT/flags.jpg"
run masks data/models/rfdetr-seg-nano-576.onnx data/dog.jpg $L --segmentation --mask-threshold -0.5 --resolution 576 --output "$OUT/masks.jpg"
run usage x y
run badthr data/models/rfdetr-nano-1101.onnx data/dog.jpg $L --threshold abc
run badint data/models/rfdetr-nano-1101.onnx data/dog.jpg $L --max-detections 1.5
run badlist data/models/rfdetr-keypoint-preview.onnx data/dog.jpg $L --keypoint --keypoint-counts 0,x
run nomodel missing.onnx data/dog.jpg $L
run gpuflag data/models/rfdetr-nano-1101.onnx data/dog.jpg $L --gpu-preprocess --output "$OUT/gpuflag.jpg"
run video data/models/rfdetr-nano-1101.onnx "$CLIP" $L --output "$OUT/vid.mp4"
cd "$OUT" && sha256sum ./*.out ./*.err ./*.rc ./*.jpg >SHA256SUMS && echo "wrote $OUT/SHA256SUMS ($(wc -l <SHA256SUMS) entries)"
