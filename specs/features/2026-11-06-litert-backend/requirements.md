# LiteRT backend requirements

Implements the new LiteRT phase in [the roadmap](../../roadmap.md). User approved the proposed scope with “go”.

## Scope

| In | Paths |
|---|---|
| CPU float32 detection, keypoint/pose estimation and segmentation behind InferenceBackend | src/backends/litert_backend.{hpp,cpp}, src/backends/litert_contract.{hpp,cpp}, src/backends/inference_backend.cpp, src/main.cpp, src/cli_options.cpp |
| Exclusive USE_LITERT build; supplied SDK prefix; staging helper | CMakeLists.txt, cmake/deps/packages/LiteRT.cmake, scripts/fetch_litert.sh |
| Direct upstream export and pinned runtime | deploy/export_litert.py, deploy/litert_keypoint_export.py, versions.env |
| Contract tests, runtime checks, CI, documentation | tests/, .github/workflows/ci.yml, README.md, docs/, AGENTS.md, specs/mission.md, specs/tech-stack.md |

Out: GPU/NPU delegates, quantized I/O, batch sizes above 1, arbitrary ONNX-converted custom operators, a new Docker image, automatic source builds of LiteRT.

## Decisions

- Preserve the interface and shared CPU preprocessing/postprocessing required by [mission](../../mission.md): no backend branches in RFDETRInference.
- Use LiteRT’s modern C API and CPU accelerator, with RAII ownership. The reference neuriplo implementation uses the older TensorFlow interpreter API; borrow its boundary validation, not its custom operators or build dependencies.
- Support the pinned RFDETR_VERSION direct `format="litert"` export: fixed square NCHW float32 input; positional boxes, class logits, optional NQHW mask logits or keypoint tensors. Reject other contracts explicitly.
- Require a staged LITERT_ROOTDIR prefix. Provide a helper using the official release SDK headers and matching ai-edge-litert wheel’s native runtime; Python is a staging/export dependency only.
- Pins live in versions.env, including LITERT_TORCH_VERSION for reproducible direct/pose export. User approval includes the new LiteRT dependency. Keep RFDETR_VERSION unchanged.

## Context

- src/backends/inference_backend.hpp:19 defines the backend interface; src/backends/inference_backend.cpp:48 constructs the selected backend.
- CMakeLists.txt:109 counts exclusive backends; backend sources and links are registered at :271 and :332.
- Unit tests must not need model files. Runtime model checks belong in integration tests and explicit manual commands.
- References: https://github.com/olibartfast/neuriplo/tree/master/backends/litert and https://github.com/olibartfast/neuriplo-infer ; the latter now delegates export recipes to neuriplo-tasks.
- Upstream contract verified at https://github.com/roboflow/rf-detr/blob/1.11.2/docs/exports/litert.md . Direct export keeps NCHW and positional outputs. Upstream documents keypoints as unsupported with the older converter; the current converter is being verified for pose. GPU remains out of scope.
- Reference coverage verified at https://github.com/olibartfast/neuriplo-infer/blob/master/docs/BackendCoverage.md (E13): native RF-DETR detection and segmentation exports pass, legacy generic onnx2tf artifacts are known-bad; its pose route uses a separate conversion; investigate whether the installed direct converter now supports the keypoint head.
- LiteRT official release: https://github.com/google-ai-edge/LiteRT/releases/tag/v2.3.0 . C headers avoid the C++ wrapper’s extra Abseil/FlatBuffers dependencies.

## Scope extension — user request on 2026-11-06

The user explicitly requires detection, pose and segmentation together in one YouTube showcase,
using the same Pexels city-street footage. This supersedes the original two-task scope.
Before enabling pose, export a real keypoint model, verify native/Python tensor parity and
visually inspect its skeletons. Prefer the existing direct LiteRT converter when it works;
no new converter dependency is authorized merely by this investigation.
The final edited deliverable is one detection/segmentation video with a synchronized comparison. The later user decision below supersedes the initial three-task showcase request.

### Verified pose export decision

The stock direct converter still fails with an invalid rank-4 batch_matmul output.
An export-only Linear adapter flattens the leading dimensions, applies the original weights,
and restores the original shape. It runs only on the exporter's private model copy and restores
the temporary exporter hook even on exceptions. Full-model eager parity is checked before
conversion at atol=rtol=1e-5. The real city frame produced zero differences for all three eager
outputs; native C++ then passed the existing 1e-3 raw keypoint gate against Python LiteRT and
the unmodified PyTorch reference. No new converter dependency was needed.

For the requested showcase, pose uses the same 8–20 second excerpt (600 frames at 50 fps),
384-square model input and class threshold 0.20. Detection/segmentation cover the full 2,208
frames at threshold 0.50. The presentation must disclose the different scopes and thresholds.

## Final showcase scope

The maintainer rejected the pose showcase after observing a false positive on a vehicle.
The final public-facing video, thumbnail, title, description and chapters include detection
and segmentation only. The experimental backend/export adapter remains available with the
measured numerical checks and explicit accuracy limitations; it is not presented as a validated
pose accuracy demonstration. The rejected edit is retained only in ignored local artifacts.
