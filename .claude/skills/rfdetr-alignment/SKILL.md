---
name: rfdetr-alignment
description: Aligns this project with an upstream roboflow/rf-detr release — verify the release against upstream notes first, diff the export and runtime contract, update pins, README and CHANGELOG. Trigger when the user says "align with rfdetr X.Y.Z", "new rfdetr release", "upstream release", "version alignment", or invokes /rfdetr-alignment.
---

# RF-DETR Upstream Alignment

The project's standing obligation ([specs/roadmap.md](../../../specs/roadmap.md) → Deferred →
Standing obligation): **every** upstream `rfdetr` release triggers an alignment pass. It is
event-driven, preempts the roadmap queue, and is one of the cases that requires a spec directory
under `specs/features/` before code is written.

Written for any coding agent — every step is a manual checklist item.

## Step 0 — The rule that is most often broken

**Never assume an upstream version is a local Git tag, and never infer the scope of work from the
version string.** A patch release can change the exported model contract; a minor release can
change nothing at all. Read `AGENTS.md`, `README.md`, and `CHANGELOG.md`, then verify the named
release against the official upstream project **before** touching anything.

Upstream: <https://github.com/roboflow/rf-detr> — read the release notes for the named tag, and
diff against the release currently recorded in `specs/tech-stack.md` (`rfdetr[onnx]`, pinned in
`deploy/requirements.txt`).

If the release does not exist upstream, stop and say so. Do not proceed on the assumption that it
will.

## Step 1 — Classify the change

Answer each of these from the upstream diff, in writing. The answers decide the whole scope.

| Question | If yes |
|----------|--------|
| Do exported model **inputs** change (shape, dtype, normalisation, resolution)? | CPU preprocessing (`src/media.cpp`) and the DALI pipelines (`data/dali/`) both move — see [specs/gpu-pipeline.md](../../../specs/gpu-pipeline.md) rules 1 and 2 |
| Do exported **outputs** change (order, count, shape, semantics)? | `validate_output_order()` in each backend, and every postprocess path, including `src/gpu/rfdetr_postprocess.cu` |
| Do required **runtime operators** change? | ExecuTorch delegate/kernel selection — see the `EXECUTORCH_BUILD_KERNELS_OPTIMIZED` note in `AGENTS.md`; ONNX opset in `docs/export.md` |
| Do the **public Python APIs** used by `deploy/` change? | `deploy/export_onnx.py`, `deploy/export_executorch.py` |
| Is it a training/dataset-only change? | Say so explicitly and state that C++ postprocessing is unaffected — that conclusion is the deliverable |

## Step 2 — CPU and GPU move together

If postprocessing changes at all: the CPU implementation and its CUDA mirror are two
implementations of **one** contract ([specs/mission.md](../../../specs/mission.md), architectural
commitments). Fixing one alone is the failure mode this rule exists to prevent. Check
`src/media.cpp`, `src/processing_utils.cpp`, and `src/gpu/rfdetr_postprocess.cu` together.

## Step 3 — Update the pins

| What | Where |
|------|-------|
| `rfdetr[onnx]` version | `deploy/requirements.txt` — the only pinned pip requirement |
| Version statement | `specs/tech-stack.md` export-tooling row, and `specs/mission.md` "currently **X.Y.Z**" |
| Export guidance | `docs/export.md` |
| Any container tag that moves with it | `scripts/fetch_dali.sh`, `scripts/generate_dali_pipelines.sh`, `export_trt.sh` — the same tag lives in all three |

## Step 4 — Documentation, mandatory

Per the Spec Sync rule in `AGENTS.md`:

- [ ] `README.md` updated in the **same change** whenever code, build options, backend versions,
      Docker images, or export packages move
- [ ] README statements verified against `CMakeLists.txt`, `CMakePresets.json`,
      `deploy/requirements.txt`, `Dockerfile`, `docs/export.md`
- [ ] `CHANGELOG.md` entry under `[Unreleased]`: a heading naming the release, a link to the
      upstream release tag, prose on what changed upstream and why it does or does not reach C++,
      and a per-file change table
- [ ] If the alignment needs **no** README change, write down in the CHANGELOG why not — that
      statement is required, not optional

## Step 5 — Verify

- [ ] Default build and unit tests: see `AGENTS.md`
- [ ] Re-export at least one model with the new package version and run it end to end
- [ ] If the TensorRT, ExecuTorch, or GPU paths are implicated, run
      [`gpu-verify`](../gpu-verify/SKILL.md) or the equivalent manual backend check — **CI tests
      none of them**
- [ ] Any behaviour that could not be verified is stated plainly in the CHANGELOG rather than
      implied to work
