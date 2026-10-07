# Phase 7 — Release hardening after v0.6.0

Implements [Phase 7](../../roadmap.md#phase-7--release-hardening-after-v060) of `specs/roadmap.md`.
Branch: `feature/phase-7-release-hardening` from `develop` at `cc0d4f9`, which is v0.6.0 (`74d008f`)
plus the release-record commit.

v0.6.0 shipped with three open items, and one older investigation is still unfinished:

1. **clang-tidy is advisory and not clean.** `lint.yml` runs clang-tidy, but `.clang-tidy` sets
   `WarningsAsErrors: ''`, so 80 findings across 8 files pass CI. `AGENTS.md` says to fix every
   finding before committing. It was not enforced, and it was not true for v0.6.0.
2. **`main()` has a cognitive complexity of 108** (threshold 25). It is the largest single
   finding: argument parsing, validation, config building, the video path, the image path and
   result printing all sit in one 300-line function.
3. **`--display` playback on the GPU path was never run.** The v0.6.0 CHANGELOG lists it as
   unverified, because the verification box had no display.
4. **The performance and memory investigation**
   ([`2026-09-22-performance-memory-investigation/`](../2026-09-22-performance-memory-investigation/))
   was stashed mid-execution before the TensorRT 11 work. Its spec required measured evidence.
   That evidence was never published, and its code (a `PROFILING` option and a CPU benchmark
   file) never merged.

## Scope

### In

| ID | Deliverable | Paths |
|----|-------------|-------|
| R-1 | Command-line parsing moves out of `main()` into a library unit with unit tests. The parsed result, error messages and exit codes are unchanged | `src/cli_options.hpp`, `src/cli_options.cpp` (new), `src/main.cpp`, `tests/unit/test_cli_options.cpp` (new), `CMakeLists.txt` |
| R-2 | `main()` is split into named steps (usage, config building, the video run, the image run, result printing), each under the cognitive-complexity threshold | `src/main.cpp` |
| R-3 | Every clang-tidy finding in the CI configuration is fixed in code, or suppressed with a `NOLINT(<check>)` and a one-line reason where the code is correct (FFmpeg's C API takes C arrays) | `src/main.cpp`, `src/rfdetr_inference.{hpp,cpp}`, `src/rfdetr_types.hpp`, `src/media.cpp`, `src/video_reader.cpp`, `src/video_writer.cpp`, `src/backends/onnx_runtime_backend.cpp` |
| R-4 | `postprocess_keypoint_outputs` (complexity 56) is split into helpers without changing a single output value | `src/rfdetr_inference.{hpp,cpp}` |
| R-5 | clang-tidy is enforced: `WarningsAsErrors: '*'`, so the `Clang-Tidy` job in `lint.yml` fails on any finding | `.clang-tidy` |
| R-6 | Behaviour is unchanged. On the fixed CLI workload, output images, stdout, stderr and exit codes are byte-identical to the v0.6.0 baseline (V-3) | — |
| R-7 | `--display` is verified on the GPU path, in the CUDA-preprocessing build and the DALI build, video and image, on this machine's RTX 3060 Laptop with a real display. The result is recorded | `validation.md`, `CHANGELOG.md` |
| R-8 | The performance and memory investigation is revived on this branch, rebased onto the post-R-1…R-4 code, and finished against its own spec: the `PROFILING` option, `bench_cpu_pipeline.cpp`, and measured results in `results.md` | `specs/features/2026-09-22-performance-memory-investigation/` (its own `requirements.md` R-1…R-16 govern), `CMakeLists.txt`, `tests/benchmark/bench_cpu_pipeline.cpp` |
| R-9 | Docs and specs follow the change: `AGENTS.md`, `specs/tech-stack.md`, `README.md`, `docs/advanced-usage.md` (the `PROFILING` option, enforced clang-tidy), and `CHANGELOG.md` `[Unreleased]` | as named |

### Out

- **Fixing the CLI's silent-ignore behaviour.** The parser drops unknown flags without a word,
  and it also ignores a value flag given as the last argument (`--threshold` with nothing after
  it). That is a behaviour change. R-6 forbids it here, and it needs its own decision → recorded
  as Open question Q-1.
- clang-tidy findings in configurations CI does not run clang-tidy on: `USE_OPENCV=ON`,
  TensorRT, ExecuTorch, `src/gpu/`. `lint.yml` runs only the default configuration, and
  `tensorrt_backend.cpp` is excluded there. Widening the lint matrix is a separate phase.
- Any optimisation that the profiling (R-8) suggests. The investigation's own spec rules out
  speedup claims without a separately measured candidate, and this phase carries that rule over.
- A release. This phase lands on `develop`, and v0.6.1/v0.7.0 is a separate `release` checklist
  run.
- Changes to `src/gpu/`, the backends' inference logic, or any numeric path. Keypoint
  postprocessing has no GPU mirror (roadmap → Deferred), so R-4 does not touch the
  "CPU and GPU move together" commitment.

## Decisions

- **D-1 — CLI parsing becomes a library unit (`rfdetr::cli`), not just helper functions in
  `main.cpp`.** Only a library unit can be unit-tested. Unit tests need no model file (mission
  commitment), and a pure argv → struct parser meets that. Backend-specific usage text
  (`kExampleModel`, `kBackendDescription`, `kBackendBuildFlags`) stays in `main.cpp`: the backend
  is a property of the executable, and the parser has no reason to know it. The GPU-flag build
  checks move into the parser. `USE_CUDA_PREPROCESS`, `USE_CUDA_POSTPROCESS` and `USE_DALI` are
  `PUBLIC` compile definitions on `rfdetr_inference_lib` (`CMakeLists.txt:390-396`), so they mean
  the same thing in the library as in `main.cpp`.
- **D-2 — Behaviour preservation is byte-level, not "looks the same".** The baseline is
  `golden_cli.sh` run against `build-baseline` (v0.6.0 + the release-record commit, ONNX
  Runtime, FFmpeg media). It covers 13 invocations and hashes stdout, stderr, the exit code and
  the output image of each, 45 hashes in all. Two baseline runs hash identically once libx264's
  pointer addresses and the output directory are normalised. Separate stdout and stderr files
  matter: replacing `std::endl` with `'\n'` changes when stdout is flushed, so a combined
  `2>&1` capture could reorder lines without any real behaviour change.
- **D-3 — Use `NOLINT`, not code changes, where the code is correct for the API it calls.** The
  FFmpeg C arrays in `video_reader.cpp` and `video_writer.cpp` (`av_strerror` buffers and the
  `sws_scale` plane/stride arrays) may become `std::array` plus `.data()` where that reads no
  worse. Otherwise they keep a scoped `// NOLINT(<check>)` with its reason. A blanket
  check-disable in `.clang-tidy` is not allowed: the user chose "enforce", not "narrow".
- **D-4 — `ModelType` gets `std::uint8_t` as its underlying type** (`performance-enum-size`). It
  is a scoped enum in a public header, and no code depends on its size or casts it to `int`
  for I/O. The implementer must grep for any `static_cast<int>(…model_type)` and report one if
  found, rather than suppress the check.
- **D-5 — The delegation follows `specs/delegation-workflow.md` in the Claude Code mapping.**
  This session is the reasoner and planner. It writes the specs and packets and does not edit
  `src/` or `tests/`. `implementer` subagents produce code, one packet each. A `reviewer`
  subagent (read-only) gives the accept/reject verdict on each diff before merge into the
  branch. `scripts/scoreboard.sh` is the acceptance command, run exactly once per packet.
- **D-6 — The perf investigation keeps its own spec directory**, restored verbatim from
  `stash@{0}` (the `46fedbc` branch tip plus the stashed revisions). It is not merged into this
  triple, so its R-1…R-16, D-1…D-11 and evidence log stay intact. This phase adds only the
  ordering: profile after R-1…R-4 land, so the measurements describe the code that will ship.
- **D-7 — `--display` verification runs in the `rfdetr-p6:*` Docker images with the host's X
  display forwarded,** not on a host build. The locally installed TensorRT 11.2.1.2 tarball
  cannot build engines (memory: Local TensorRT gap), while the images carry the matching NGC
  stack.

## Context

- Existing patterns: `src/video_reader.hpp` and `src/video_writer.hpp` (pimpl, `rfdetr::` sub-namespaces).
  Unit tests follow `tests/unit/test_rfdetr_inference.cpp` and are registered via
  `UNIT_TEST_SOURCES` (`CMakeLists.txt:426`). Library sources go in `RFDETR_SOURCES`
  (`CMakeLists.txt:242`).
- Current parser semantics, which must be preserved exactly (`src/main.cpp:99-260`):
  - `argc < 4` prints the usage text and returns 1.
  - Optional flags are scanned from index 4. Unknown tokens are ignored, and a value flag in
    the last position is ignored.
  - Value errors print `Error: <flag> expects an integer|a number|comma-separated integers, got '<v>'`
    and return 1.
  - Range checks run after the scan: threshold in [0, 1], resolution > 0, max-detections > 0,
    `--gpu-postprocess` requires `--segmentation`.
  - The build-gated GPU-flag checks run last.
  - `--background-class-id none` sets the "given" flag and leaves the value empty.
  - The first validation that fails prints its message to stderr and returns 1.
- Constraints carried in: exactly one backend; unit tests need no model; GPU tests skip; the
  `AGENTS.md` pre-commit gate (format, clang-tidy, cppcheck) applies to every commit on this
  branch. Once R-5 lands, clang-tidy fails the build outright.
- `perf_event_paranoid` on this host is `4`. R-8 needs the user to lower it for the session, as
  the investigation's D-7 records. The agent cannot do that itself.
- The host is not idle (the investigation's validation records a k3d cluster and AnyDesk). R-8
  measurements follow its quiet-host and `taskset` protocol.

## Open questions

- **Q-1** — Should unknown flags and a value flag missing its value become errors? That is a
  user-visible behaviour change and is deferred out of this phase. It is recorded so the R-1
  parser's tests pin today's behaviour on purpose, not by accident.
- **Q-2** — R-7 needs the user's `DISPLAY` and X authorisation (`xhost +local:docker` or an
  Xauthority mount) at run time. If the window opens but cannot be watched, R-7 is recorded as
  "window opened, not inspected" and not as a pass.
