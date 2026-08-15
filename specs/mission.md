# Mission

RF-DETR C++ Inference runs the RF-DETR model — object detection, instance segmentation, and keypoint estimation — as a native C++20 command-line application, with no Python in the loop at inference time.

The point is portability across runtimes. The same source builds against ONNX Runtime, TensorRT, or ExecuTorch, chosen at compile time, so the same postprocessing and drawing code serves a CPU laptop, an NVIDIA server, and an edge device. It handles single images and video, the latter through a multi-threaded ring-buffer pipeline, with an opt-in GPU pipeline (DALI preprocessing + CUDA segmentation postprocessing) on the TensorRT backend.

The project is kept deliberately in step with upstream [`rfdetr`](https://github.com/roboflow/rf-detr) releases — currently **1.9.2**. Every upstream release triggers an alignment pass, recorded in [CHANGELOG.md](../CHANGELOG.md).

## Architectural commitments

These are the invariants. Breaking one is a design change, not a bug fix.

- **Exactly one inference backend compiles in.** Enabling two is a configure-time error (`CMakeLists.txt:93-108`). This became an error in v0.4.0; before that the build silently fell back to ONNX Runtime.
- **Backends live behind an interface.** Every backend implements `rfdetr::backend::InferenceBackend` (`src/backends/inference_backend.hpp:19`) and is constructed by `create_backend()`. Adding or changing a backend means working inside that interface — never adding backend branches to `RFDETRInference`.
- **Media and display are swappable the same way.** `USE_OPENCV` selects OpenCV or FFmpeg+SDL2+stb behind `src/media.hpp` and `src/display.hpp`. No backend-specific type may leak into core code.
- **CPU and GPU postprocessing stay numerically in step.** They are two implementations of one contract. The v0.4.0 bilinear-resize fix landed in `src/media.cpp` *and* was mirrored into `src/gpu/rfdetr_postprocess.cu` for exactly this reason. Never fix one alone.
- **Unit tests need no model file.** A backend is injected through the second `RFDETRInference` constructor together with `tests/unit/mock_backend.hpp`.
- **GPU tests skip, they do not fail.** GPU-dependent tests call `GTEST_SKIP()` when `cudaGetDeviceCount` reports zero, so CI can compile GPU targets on runners without a GPU.

## Components

| Path | Role |
|------|------|
| `src/main.cpp` | CLI entry point; parses flags, routes image vs. video by file extension |
| `src/rfdetr_inference.*` | Orchestrator — `Config`, `ModelType`, preprocess → infer → postprocess → draw |
| `src/backends/` | `InferenceBackend` interface, the three implementations, and the factory |
| `src/media.*` | Image/mask types, load/save, CPU preprocessing, drawing and text |
| `src/display.*` | Live preview window (`--display`), no-op when headless |
| `src/video_reader.*`, `src/video_writer.*` | Decode/encode behind pimpl |
| `src/video_pipeline.*` | Four-stage `std::jthread` pipeline over a bounded ring buffer |
| `src/processing_utils.*` | Pure helpers: sigmoid, normalize, box convert/scale/clamp |
| `src/gpu/` | Opt-in GPU pipeline: CUDA context, DALI preprocessor, segmentation kernels |
| `cmake/deps/` | Dependency-resolution facade over apt / conan / vcpkg |
| `deploy/` | Python export tooling (`.onnx`, `.pte`, DALI pipelines) |
| `scripts/` | DALI staging and pipeline generation |

## Out of scope

Deliberately not built, each for a recorded reason — see [roadmap.md](roadmap.md#deferred):

- Batch size > 1. Every tensor contract fixes batch 1.
- GPU postprocessing for detection and keypoint. Detection postprocess is not a bottleneck; keypoint postprocess is branch-heavy and belongs on the CPU until profiling says otherwise.
- GPU rendering. Drawing stays in `src/media.cpp`.
- ONNX Runtime execution providers. The backend registers none, so ONNX Runtime is CPU-only here even with a CUDA build.

## Where to go next

- [tech-stack.md](tech-stack.md) — pinned versions, CMake options, CI coverage
- [roadmap.md](roadmap.md) — the phased work queue and the deferred list
- [gpu-pipeline.md](gpu-pipeline.md) — GPU design constraints; the model contract to check any `src/gpu/` change against
- [features/](features/) — the spec directory for each phase of work
- [AGENTS.md](../AGENTS.md) — build, test, and lint commands, and the workflow
- [README.md](../README.md) — full user-facing reference
