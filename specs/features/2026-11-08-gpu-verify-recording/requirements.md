# Requirements — GPU verification record (v0.7.1 gate)

The [gpu-verify checklist](../../AGENTS.md#checklist-gpu-verify) is the only thing that exercises
TensorRT, CUDA and DALI before a release, because CI runners have no GPU. This spec records the
hardware run that backs the v0.7.1 release, in the place
[`specs/tech-stack.md`](../../tech-stack.md) designates for it.

## Scope

| In | Path |
|----|------|
| Gate driver that runs the executable part of the checklist unattended | `scripts/run_gate.sh` |
| Container harness that gives the gate the pinned toolchain | `scripts/gate_in_container.sh` |
| Recorded results, environment and residuals | `validation.md` |

**Out:** any change to `src/`, `data/dali/` or the tolerances themselves. This is a measurement
pass, not a code change.

## Decisions

| Decision | Reason |
|----------|--------|
| Run the build and the app inside `nvcr.io/nvidia/tensorrt:<NGC_CONTAINER_TAG>-py3` | The image is the same base `dockerfile.trt` uses, so TensorRT, CUDA and DALI are the pinned ones rather than whatever the host OS ships. A bare-metal attempt failed on a toolkit/driver mismatch that the image does not have. |
| Upgraded the rented box to driver 580 | TensorRT 11.2.1.2 is built for CUDA 13.x; the instance shipped driver 570 (CUDA 12.8 ceiling), and every CUDA call failed with "requires a CUDA device". |
| Reused the staged DALI from `fetch_dali.sh` | Avoids a second 25 GB pull; the staging layout is identical to the image's. |
| `SKIP_BUILD_MATRIX=1` on the container run | The independent-halves builds and the four configure guards had already passed on the same commit in the bare-metal attempt, and `gpu-compile.yml` builds them on every PR. |

## Context

- Model: `rfdetr-seg-medium` at 432, exported with `deploy/export_segmentation.py` from
  `rfdetr[onnx]` 1.12.0.
- Video: 1000 frames, 768x576, 25 fps.
- The gate reports an unrun check as `UNRUN`, never as passing
  ([`specs/mission.md`](../../mission.md)).
