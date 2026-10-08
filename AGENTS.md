# AGENTS.md

## Project Specs
Read these before starting work; this file covers commands, the specs cover intent.
- [specs/mission.md](specs/mission.md) — what this project is and the architectural commitments not to break
- [specs/tech-stack.md](specs/tech-stack.md) — pinned versions and where each pin lives, CMake options, CI coverage
- [specs/roadmap.md](specs/roadmap.md) — phased work queue and deferred items
- [specs/gpu-pipeline.md](specs/gpu-pipeline.md) — GPU design constraints; the 8-rule model contract is the review checklist for any change to `src/gpu/`
- [specs/features/](specs/features/) — one directory per phase of work: `requirements.md`, `plan.md`, `validation.md`
- [specs/rented-gpu-runbook.md](specs/rented-gpu-runbook.md) — the operational half of the [gpu-verify checklist](#checklist-gpu-verify): renting a box, running `scripts/run_gate.sh` on it unattended, collecting results
- [specs/delegation-workflow.md](specs/delegation-workflow.md) — the harness-agnostic reasoner/planner/implementer delegation setup (OpenCode, Claude Code, Codex CLI, or anything else that can host the four primitives); it sits beside this file's workflow, it does not replace it

`docs/` is for users of the inference application; instructions aimed at whoever (or whatever) *works on* this repository live here in `specs/` and in this file's workflow checklists.

## Workflow
The loop: pick the next unticked phase in `specs/roadmap.md` → write its spec → implement from `plan.md` → pass `validation.md` → update `CHANGELOG.md` → merge to `develop` → tick the phase.

A spec directory under `specs/features/YYYY-MM-DD-<name>/` is **required** for a roadmap phase, a release, an upstream `rfdetr` alignment, or any change touching a path CI cannot execute (`src/gpu/`, `src/backends/tensorrt_backend.cpp`, `src/backends/executorch_backend.cpp`, `deploy/export_executorch.py`). It is **not** required for bug fixes, docs, or dependency bumps with no contract change — those go in `CHANGELOG.md` only. "CHANGELOG only" is about specs, not a mandate to log everything: `CHANGELOG.md` covers this C++ project's own features and fixes and the upstream `rfdetr` releases it tracks, so changes confined to agent tooling (`specs/`, `.claude/`, `.opencode/`, `AGENTS.md` and its pointer files) are recorded in the commit message alone.

Four workflows are written down as checklists at the end of this file. Each is plain markdown, usable by hand with any agent:

| Checklist | Use when |
|-----------|----------|
| [feature-spec](#checklist-feature-spec) | Starting a roadmap phase — find it, branch, interview, write the spec triple |
| [rfdetr-alignment](#checklist-rfdetr-alignment) | An upstream `rfdetr` release lands — the standing obligation |
| [release](#checklist-release) | Cutting a git-flow release |
| [gpu-verify](#checklist-gpu-verify) | Verifying TensorRT/DALI/CUDA — the gate CI cannot run |

Git-flow: branch from `develop`, never from `master`.

## Backend Selection
Exactly one backend is compiled in; enabling two is a configure-time error.
- ONNX Runtime (default): 
  `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- TensorRT:
  `cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- ExecuTorch (`.pte` models, rfdetr 1.9.0+):
  `cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_EXECUTORCH=ON -DEXECUTORCH_ROOTDIR=<prefix> -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
  - `-DEXECUTORCH_DELEGATE=xnnpack|portable` (default `xnnpack`) must match the delegate the `.pte` was exported with.
  - Without `EXECUTORCH_ROOTDIR` the build falls back to compiling the pinned ExecuTorch (`EXECUTORCH_VERSION`) from source, which is slow and needs a Python interpreter with ExecuTorch's build deps (`import torchgen`, i.e. the `torch` wheel — a bare `python3` fails).
  - The prefix must be built with `-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` (defaults to `OFF`): `.pte` files from rfdetr 1.9.1+ call `aten::linear.out`, which only `optimized_native_cpu_ops_lib` registers. The build links that lib when present and warns + falls back to `portable_ops_lib` when not — exactly one op library, since duplicate kernel registration aborts at startup. See [docs/building.md](docs/building.md#building-the-executorch-install-prefix), "Building the ExecuTorch install prefix".
  - The `extension/evalue_util` install-path patch is only needed on v1.3.1 and older; v1.4.0 fixed it upstream.

## Docker
Three backend Dockerfiles build the inference-backend × media-backend matrix; the backend is
chosen by which file you pass to `-f` (there is no bare `Dockerfile`):
- `dockerfile.onnxrt` — ONNX Runtime (CPU), `--build-arg MEDIA_BACKEND=ffmpeg|opencv`
- `dockerfile.executorch` — ExecuTorch (CPU, `.pte`), `--build-arg MEDIA_BACKEND=ffmpeg|opencv`;
  builds the ExecuTorch runtime from source into `/opt/executorch` (override the tag with
  `--build-arg EXECUTORCH_VERSION=<tag>`) and applies the upstream install fix automatically
- `dockerfile.trt` — TensorRT (GPU), `--build-arg MEDIA_BACKEND=ffmpeg|opencv` and
  `--build-arg GPU_PIPELINE=off|pre|post|on|dali|dali-on` (GPU pipeline: `on` = CUDA pre + post,
  `dali-on` = DALI pre + CUDA post; the retired `cuda` value fails the build)
- The blocks shared across all three are wrapped in `# === shared:<name> ===` markers and
  guarded by `./scripts/check_dockerfile_parity.sh` (the `Dockerfile shared blocks` step in `lint.yml`).
- **Pre-commit Docker gate:** before committing a change to `dockerfile.*` or a Docker-coupled
  build option, dependency pin, or script, build every affected Dockerfile/argument combination
  that the current machine and environment can support. A TensorRT image build does not itself
  require a GPU, so run it whenever Docker, NGC access, network, architecture, disk, and memory
  are available; when suitable NVIDIA hardware is present, also run the affected image with
  `--gpus all` using the [gpu-verify checklist](#checklist-gpu-verify). Record unavailable cases
  as `UNRUN` with the exact reason, never as passing, and do not commit while any locally runnable
  affected build fails.

## GPU Pipeline (TensorRT only)
- Build both halves (CUDA preprocessing with nvJPEG + CUDA postprocessing; needs nvcc and the toolkit's nvJPEG):
  `cmake -S . -B build -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON -DUSE_GPU_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- DALI is the alternative GPU preprocessor. Stage it first (one-time, extracts from pinned Triton container): `./scripts/fetch_dali.sh` → `~/dependencies/dali`, then add `-DUSE_DALI=ON -DDALI_ROOT=$HOME/dependencies/dali` to the command above
- Halves are independent. Preprocessing is `-DUSE_CUDA_PREPROCESS=ON` (kernel + nvJPEG, needs nvcc) **or** `-DUSE_DALI=ON` (no nvcc); both together is a configure-time `FATAL_ERROR`. Postprocessing is `-DUSE_CUDA_POSTPROCESS=ON` (CUDA seg postprocessing, needs nvcc). `CMAKE_CUDA_ARCHITECTURES` defaults to `86`
- Either option with the ONNX Runtime backend is a configure-time `FATAL_ERROR`
- Runtime flags (default off): `--gpu-preprocess` (whichever preprocessor was compiled in), `--gpu-postprocess` (segmentation only), `--dali-pipeline-dir <dir>` (DALI builds only, default `data/dali`)
- DALI only: regenerate `.dali` pipelines for a new resolution with `./scripts/generate_dali_pipelines.sh <res>` (needs `--gpus all` Docker); 432 and 576 are checked in. The CUDA preprocessor runs at any resolution
- GPU unit tests (`test_gpu_postprocess.cpp`, `test_gpu_parity.cpp`) `GTEST_SKIP()` without a CUDA device; like TensorRT, CI compiles but does not execute GPU paths — `gpu-compile.yml` builds TensorRT alone, each preprocessor (CUDA, DALI), CUDA postprocessing and both full pipelines with `-DWERROR=ON` against headers staged by `scripts/ci/stage_gpu_headers.sh`, so a compile break is a red PR, not a surprise on metered hardware. Behaviour still has to be tested manually with the [gpu-verify checklist](#checklist-gpu-verify)
- On a rented GPU box, `./scripts/run_gate.sh` drives the executable part of that checklist unattended and reports the rest as `UNRUN`; it arms a deadline watchdog and stops the instance when done. Env knobs: `CUDA_ARCH` (default: the card's compute capability), `DEADLINE_HOURS`, `SKIP_DEFAULT_PATH`, `SKIP_BUILD_MATRIX`, `SELF_STOP`, `MODEL`, `VIDEO`. On Google Colab (other GPU generations, no Docker), use `scripts/colab/gpu_gate.ipynb`; DALI is then staged with `DALI_SOURCE=pip ./scripts/fetch_dali.sh`. End-to-end procedure — choosing an instance, export prep, setup script, collecting results: [specs/rented-gpu-runbook.md](specs/rented-gpu-runbook.md)
- Design constraints: [specs/gpu-pipeline.md](specs/gpu-pipeline.md) — remaining phases: [specs/roadmap.md](specs/roadmap.md)

## Dependency Versions

Export package: `rfdetr[onnx]` at `RFDETR_VERSION`; every pin is in `versions.env`, and prose names the variable, not the value.
TensorRT: pinned 11.x (`TENSORRT_VERSION`, the `nvcr.io/nvidia/tensorrt:<NGC_CONTAINER_TAG>-py3` stack) and DALI 2.x (`DALI_VERSION`, the alternative GPU preprocessor; the default CUDA preprocessor's nvJPEG comes with the `CUDA_VERSION` toolkit and has no pin). One version each: no older-release branches, no second compile-check pin — `tensorrt_backend.hpp` rejects `NV_TENSORRT_MAJOR < 11`.
**[`versions.env`](versions.env) is the single source of truth for every third-party pin.** Never
hardcode a version anywhere else.
- CMake reads it via `cmake/versions.cmake` (included before `cmake/deps/Deps.cmake`); each pin is a `CACHE STRING`, so `-DTENSORRT_VERSION=…` overrides it.
- Shell scripts read it via `source scripts/versions.sh`, which never clobbers a value already in the environment — `TRITON_IMAGE=… ./scripts/fetch_dali.sh` still works.
- Some coordinates are derived, not stored — do not add variables for them. `cmake/versions.cmake` derives `TENSORRT_SHORT_VERSION` only (all CMake needs). `scripts/versions.sh` derives that plus `TENSORRT_DEB_VERSION`, `TRITON_IMAGE` and `TENSORRT_IMAGE`, which no CMake consumer uses.
- Five formats cannot read a file — the backend Dockerfiles' `ARG` defaults, `conanfile.txt`, `deploy/requirements.txt`, the argparse defaults in `deploy/export_*.py`, and the two version tables in `README.md`. They restate the values; `./scripts/check_version_sync.sh` (the `Version Sync` job in `lint.yml`) fails when a restatement drifts.
- **No other prose states a pinned value.** `docs/`, `specs/` and the rest of the README name the `versions.env` variable (`TENSORRT_VERSION`, …); commands read it with `source scripts/versions.sh`. Historical statements ("fixed in v1.4.0") are not pins and stay.
- **After editing `versions.env`, run `./scripts/check_version_sync.sh`** and fix what it reports; no hand reconciliation of prose is needed.

## Dependency Resolution
- ONNX Runtime automatic downloads cover Linux/Windows x64/arm64. Other targets may supply a compatible prefix or package-manager build; loading the catalog with ONNX disabled must not reject them.
- Default (`-DDEPS_MODE=apt`): system packages + pinned downloads — no extra tooling
- Conan/vcpkg: auto-activate via toolchain; see [docs/package-manager-architecture.md](docs/package-manager-architecture.md)
- `-DDEPS_DEBUG=ON` logs which handler resolved each dependency

## Code Quality
- **Mandatory pre-commit / pre-push gate:** before every `git commit` and again before every `git push`, run the cppcheck and lint commands below (clang-format check, clang-tidy, cppcheck — the same checks as the `lint.yml` jobs) and fix every finding. Do not commit or push while any of them fails. If a tool is not installed or cannot run on the current machine, record it as `UNRUN` with the exact reason in the commit message or PR description, never as passing. `pre-commit run --all-files` covers clang-format and cppcheck but **not** clang-tidy, so it does not satisfy this rule on its own.
- Version pin sync: `./scripts/check_version_sync.sh`
- Format check: `find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror`
- Format apply: `find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 -i`
- Clang-tidy: 
  `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
  `find src -name '*.cpp' ! -name 'tensorrt_backend.cpp' | xargs clang-tidy-18 -p build` (same exclusion as `lint.yml`; the default configure has no TensorRT headers)
- Cppcheck: `cppcheck --enable=all --std=c++20 --suppress=missingIncludeSystem --suppress=unmatchedSuppression --suppress=unusedFunction --error-exitcode=1 -I src src/`
- Strict warnings (CI): `-DWERROR=ON` at configure time

## Spec Sync
- Mandatory: `git fetch` and check the branch against its upstream **before starting any work** — not just before pushing. A stale local branch hides the rules you are supposed to follow: on 2026-08-24 an alignment pass was written against a `develop` four commits behind `origin/develop`, missing an already-completed 1.9.2 alignment, the `specs/` tree, and the `rfdetr-alignment` workflow that governs the task. If the branch is behind, integrate or at least read what landed, then re-read `AGENTS.md` and `specs/` before planning.
- Mandatory: before acting on any release, version-alignment, or dependency-sync request, read `AGENTS.md`, `README.md`, and `CHANGELOG.md`, then verify the named release against the official upstream project. Never assume that an upstream version is a local Git tag or infer the required scope from the version string alone; inspect the repository documentation and upstream release notes/diff first.
- A change to `specs/mission.md` or `specs/tech-stack.md` must propagate in the **same commit** to `README.md`, `AGENTS.md`, and any open spec under `specs/features/`. The constitution and what it describes never diverge across commits.
- Mandatory for every release or dependency-facing patch: update `README.md` in the same change when code, build options, backend versions, Docker images, or Python export packages change.
- README version tables are verified against `versions.env` by `./scripts/check_version_sync.sh`; check the rest of the README (build options, backend constraints) against `CMakeLists.txt`, `CMakePresets.json`, `dockerfile.*`, and `docs/export.md`.
- README must list current C++ library/runtime versions, CMake options, backend constraints, and pip
  packages used for export tooling. `README.md` is the quick start and carries these at a glance (the
  `Versions at a Glance`, `Common Build Options` and `Choosing a Backend` sections); the exhaustive
  reference — every CMake option, the per-backend constraints, the ONNX Runtime archive table — lives in
  `docs/advanced-usage.md`. Both must be updated together, and the README must keep linking to it.
- If a release intentionally needs no README change, say why in `CHANGELOG.md` or the PR/release notes.

## Testing
- Unit tests: `ctest --test-dir build --output-on-failure -R UnitTests`
- Integration tests: `ctest --test-dir build --output-on-failure -R IntegrationTests`
- All tests: `cmake --build build --target run_tests`
- Benchmarks (if enabled): `./build/benchmarks`

## Sanitizers
ASan+UBSan, strict UBSan, and TSan are mutually exclusive (pick one).

### AddressSanitizer + UndefinedBehaviorSanitizer
- Configure: `cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DSANITIZERS=ON`
- Build: `cmake --build build-san --parallel`
- Run unit tests: `./build-san/unit_tests`
- Run integration tests: `./build-san/integration_tests`

### Strict UndefinedBehaviorSanitizer
- Configure: `cmake -S . -B build-strict-ubsan -DCMAKE_BUILD_TYPE=Debug -DSTRICT_UBSAN=ON`
- Build: `cmake --build build-strict-ubsan --parallel`
- Run unit tests: `./build-strict-ubsan/unit_tests`
- Run integration tests: `./build-strict-ubsan/integration_tests`

### ThreadSanitizer (data races)
- Configure: `cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DTHREAD_SANITIZER=ON`
- Build: `cmake --build build-tsan --parallel`
- Run: `TSAN_OPTIONS="halt_on_error=1" ./build-tsan/unit_tests`

## Valgrind / Profiling
Requires a plain Debug build (no sanitizers — ASan/TSan conflict with Valgrind). The `memcheck`, `callgrind`, and `massif` CMake targets are auto-generated when Valgrind is found.
- Configure: `cmake -S . -B build-valg -DCMAKE_BUILD_TYPE=Debug`
- Memcheck (correctness — run by CI): `cmake --build build-valg --target memcheck`
- CPU/cache profile: `cmake --build build-valg --target callgrind` → read with `callgrind_annotate build-valg/callgrind.out.<pid>`
- Heap profile: `cmake --build build-valg --target massif` → read with `ms_print build-valg/massif.out.<pid>`
- Profilers run on `benchmarks` if built (`-DBENCHMARKS=ON`), else `inference_app` (pass args via `-DVALGRIND_PROFILE_ARGS="..."`).
- Lower-overhead alternative: `perf record ./build/benchmarks && perf report`.
- Optional suppressions file: `valgrind.supp` at repo root is picked up automatically if present.

## Pre-commit
- Install: `pip install pre-commit && pre-commit install`
- Run all: `pre-commit run --all-files`

## Usage
- Detection: `./build/inference_app model.onnx image.jpg coco-labels-91.txt`
- Segmentation: add `--segmentation`
- Video: replace image with video file (e.g., video.mp4)
- Display: add `--display`
- TensorRT engine: use .engine or .trt model file

## Notes
- Only one backend (ONNX Runtime, TensorRT, or ExecuTorch) can be enabled at compile time.
- TensorRT requires manually installed CUDA toolkit.
- Data directory is auto-created by CMake.
- CI compiles the TensorRT backend and the GPU pipeline (`gpu-compile.yml`) but cannot run them; ExecuTorch is neither compiled nor run by CI. Test the behaviour of all three manually.

## Workflow Checklists
The four workflows from the [Workflow](#workflow) table. Each was a `.claude/skills/<name>/SKILL.md` file; they live here now so every agent and human reads them from the one file.

### Checklist: feature-spec

*Starts a phase of work by finding the next unticked phase in specs/roadmap.md, creating a branch, interviewing the user about scope/decisions/context, and writing specs/features/YYYY-MM-DD-<name>/{requirements,plan,validation}.md. Trigger when the user says "feature spec", "next phase", "start the next roadmap phase".*

#### When a spec is required

**Required** — a roadmap phase, a release, an upstream `rfdetr` alignment, or any change touching a
path CI cannot execute: `src/gpu/`, `src/backends/tensorrt_backend.cpp`,
`src/backends/executorch_backend.cpp`, `deploy/export_executorch.py`.

**Not required** — bug fixes, documentation, dependency bumps with no contract change. Those are
recorded in `CHANGELOG.md` only. Do not manufacture a spec directory for a two-line fix.

#### Workflow

##### 1. Find the next phase

Read `specs/roadmap.md`. The next phase is the **first section whose items are all `[ ]`**. Note
its number and name — they become the branch name and the spec directory name.

If the user named a specific phase instead, use that one and say which.

##### 2. Create the branch

Git-flow: branch from `develop`, never from `master`.

```bash
git checkout develop && git pull
git checkout -b feature/phase-<N>-<kebab-name>
```

##### 3. Interview the user — BEFORE writing any file

Ask exactly **three** grouped questions, in one exchange. Do not write to disk until all three are
answered.

| Group | What to ask about |
|-------|-------------------|
| **Scope** | What is in, what is explicitly out, which files and targets are touched |
| **Decisions** | The choices that would otherwise be made silently — data format, where a fixture lives, whether a change is allowed to touch `src/`, tolerance values |
| **Context** | Constraints shaping the work — hardware available for verification, CI limits, related open items, anything upstream |

For a phase whose detail is already fully written in `specs/roadmap.md`, say so and confirm rather
than re-asking; the interview exists to surface undecided things, not to re-read the roadmap aloud.

##### 4. Read the constitution before drafting

Always: `specs/mission.md`, `specs/tech-stack.md`, and the phase's own roadmap section.
Additionally, when the phase touches `src/gpu/` or `data/dali/`: `specs/gpu-pipeline.md` — its
8-rule model contract is the review checklist for anything GPU.

Also read `AGENTS.md` for the exact build, test, lint, and sanitizer commands. Never invent a
command; quote the one that is written down.

##### 5. Write the spec directory

`specs/features/YYYY-MM-DD-<feature-name>/` using today's date.

**`requirements.md`**
- *Scope* — an "In" table of deliverables with real paths, and an explicit "Out" list. Name the
  phase it implements and link back to `specs/roadmap.md`.
- *Decisions* — each choice with the reason. Where a decision is forced by an architectural
  commitment in `specs/mission.md`, cite it.
- *Context* — existing patterns to follow with `path:line` references, constraints carried in from
  the constitution, and any open question to resolve during implementation.

**`plan.md`**
- Numbered task groups, each independently implementable, each stating whether it needs a GPU.
- Sub-tasks numbered continuously across groups, with the file each one creates or edits and the
  `CMakeLists.txt` line where it must be registered.

**`validation.md`**
- *Automated — no GPU*: the commands from `AGENTS.md` that must pass on CI.
- *Automated — with a device*: what only real hardware can prove, with explicit tolerances.
- *Compile-without-device*: for anything GPU, that the target still builds and the tests report
  `SKIPPED` rather than `FAILED`.
- *Manual*: what a human has to look at.
- *Definition of done*: CHANGELOG updated, roadmap items ticked, branch merged and deleted.

Every tolerance must be a number. "Close enough" is not a gate.

#### Constraints

- No new dependency without user approval — `specs/tech-stack.md` owns the pins.
- Respect the architectural commitments in `specs/mission.md`; in particular, unit tests need no
  model file, and GPU tests skip rather than fail.
- Keep the phase independently shippable and the tree building at every step.

#### Closing a phase

When `validation.md` is fully ticked: update `CHANGELOG.md` under `[Unreleased]` in the house style
(prose plus a per-file table), tick the phase's items in `specs/roadmap.md` and mark the heading
`(Complete)`, then merge into `develop` and delete the branch.

### Checklist: rfdetr-alignment

*Aligns this project with an upstream roboflow/rf-detr release — verify the release against upstream notes first, diff the export and runtime contract, update pins, README and CHANGELOG. Trigger when the user says "align with rfdetr X.Y.Z", "new rfdetr release", "upstream release", "version alignment".*

The project's standing obligation ([specs/roadmap.md](specs/roadmap.md) → Deferred →
Standing obligation): **every** upstream `rfdetr` release triggers an alignment pass. It is
event-driven, preempts the roadmap queue, and is one of the cases that requires a spec directory
under `specs/features/` before code is written.

#### Step 0 — The rule that is most often broken

**Never assume an upstream version is a local Git tag, and never infer the scope of work from the
version string.** A patch release can change the exported model contract; a minor release can
change nothing at all. Read `AGENTS.md`, `README.md`, and `CHANGELOG.md`, then verify the named
release against the official upstream project **before** touching anything.

Upstream: <https://github.com/roboflow/rf-detr> — read the release notes for the named tag, and
diff against the release currently recorded in `specs/tech-stack.md` (`rfdetr[onnx]`, pinned in
`deploy/requirements.txt`).

If the release does not exist upstream, stop and say so. Do not proceed on the assumption that it
will.

#### Step 1 — Classify the change

Answer each of these from the upstream diff, in writing. The answers decide the whole scope.

| Question | If yes |
|----------|--------|
| Do exported model **inputs** change (shape, dtype, normalisation, resolution)? | CPU preprocessing (`src/media.cpp`) and the DALI pipelines (`data/dali/`) both move — see [specs/gpu-pipeline.md](specs/gpu-pipeline.md) rules 1 and 2 |
| Do exported **outputs** change (order, count, shape, semantics)? | `validate_output_order()` in each backend, and every postprocess path, including `src/gpu/rfdetr_postprocess.cu` |
| Do required **runtime operators** change? | ExecuTorch delegate/kernel selection — see the `EXECUTORCH_BUILD_KERNELS_OPTIMIZED` note in `AGENTS.md`; ONNX opset in `docs/export.md` |
| Do the **public Python APIs** used by `deploy/` change? | `deploy/export_onnx.py`, `deploy/export_executorch.py` |
| Is it a training/dataset-only change? | Say so explicitly and state that C++ postprocessing is unaffected — that conclusion is the deliverable |

#### Step 2 — CPU and GPU move together

If postprocessing changes at all: the CPU implementation and its CUDA mirror are two
implementations of **one** contract ([specs/mission.md](specs/mission.md), architectural
commitments). Fixing one alone is the failure mode this rule exists to prevent. Check
`src/media.cpp`, `src/processing_utils.cpp`, and `src/gpu/rfdetr_postprocess.cu` together.

#### Step 3 — Update the pins

| What | Where |
|------|-------|
| `rfdetr[onnx]` version | `deploy/requirements.txt` — the only pinned pip requirement |
| Version statement | `RFDETR_VERSION` in `versions.env` (restated in `deploy/requirements.txt` and the README tables; `check_version_sync.sh` verifies both) |
| Export guidance | `docs/export.md` |
| Any container tag that moves with it | `scripts/fetch_dali.sh`, `scripts/generate_dali_pipelines.sh`, `export_trt.sh` — the same tag lives in all three |

#### Step 4 — Documentation, mandatory

Per the Spec Sync rule in `AGENTS.md`:

- [ ] `README.md` updated in the **same change** whenever code, build options, backend versions,
      Docker images, or export packages move
- [ ] `RFDETR_VERSION` bumped in `versions.env` and `./scripts/check_version_sync.sh` passes (it
      covers `deploy/requirements.txt` and the README tables; other prose names the variable)
- [ ] Remaining README statements verified against `CMakeLists.txt`, `CMakePresets.json`,
      `dockerfile.*`, `docs/export.md`
- [ ] `CHANGELOG.md` entry under `[Unreleased]`: a heading naming the release, a link to the
      upstream release tag, prose on what changed upstream and why it does or does not reach C++,
      and a per-file change table
- [ ] If the alignment needs **no** README change, write down in the CHANGELOG why not — that
      statement is required, not optional

#### Step 5 — Verify

- [ ] Default build and unit tests: see `AGENTS.md`
- [ ] Re-export at least one model with the new package version and run it end to end
- [ ] If the TensorRT, ExecuTorch, or GPU paths are implicated, run
      [`gpu-verify`](#checklist-gpu-verify) or the equivalent manual backend check — **CI tests
      none of them**
- [ ] Any behaviour that could not be verified is stated plainly in the CHANGELOG rather than
      implied to work

### Checklist: release

*Cuts a git-flow release — Spec Sync documentation checks, [Unreleased] to [vX.Y.Z] in CHANGELOG.md, version-statement reconciliation across CMakeLists.txt/vcpkg.json/README, then release branch, tag, and merge back. Trigger when the user says "cut a release", "release vX.Y.Z", "prepare the release".*

#### Step 0 — Read before acting

Mandatory, in this order: `AGENTS.md`, `README.md`, `CHANGELOG.md`, `specs/roadmap.md`. If the
release includes an upstream `rfdetr` alignment, verify that release against
<https://github.com/roboflow/rf-detr> — never assume an upstream version is a local Git tag. See
[`rfdetr-alignment`](#checklist-rfdetr-alignment).

#### Step 1 — Confirm the gate

`specs/roadmap.md` says which phases the release is gated on. Do not cut a release with an unticked
gating phase unless the user explicitly decides to — and if they do, record that decision in the
CHANGELOG.

For anything touching TensorRT, ExecuTorch, DALI, or CUDA: **CI has never built or run it.** Run
[`gpu-verify`](#checklist-gpu-verify) and the manual backend checks first, or state in the release
notes exactly what went out unverified.

#### Step 2 — Reconcile the version statements

The project version is stated in four places, and they must agree (the first three were reconciled
in v0.5.0):

| Location | Form |
|----------|------|
| `CMakeLists.txt` | `project(rfdetr_inference VERSION X.Y.Z ...)` |
| `vcpkg.json` | `"version-string"` |
| `README.md` | the version badge and its release-tag link |
| `specs/tech-stack.md` | the "`project()` declares `VERSION X.Y.Z`" sentence |

Set all four to the release version in the release commit. Dependency
pins live only in `versions.env`: run `./scripts/check_version_sync.sh`, which verifies every
restatement (Dockerfile `ARG`s, `conanfile.txt`, `deploy/requirements.txt`, export defaults, the
README version tables), and see "Known pin duplications" in `specs/tech-stack.md`.

#### Step 3 — Spec Sync checklist

From `AGENTS.md`. Every box is mandatory:

- [ ] `README.md` updated in the same change as any code, build-option, backend-version, Docker, or
      export-package move
- [ ] `./scripts/check_version_sync.sh` passes (README version tables and every other restatement)
- [ ] README build options and backend constraints verified against `CMakeLists.txt`,
      `CMakePresets.json`, `dockerfile.*`, `docs/export.md`
- [ ] README lists current C++ library/runtime versions, CMake options, backend constraints, and
      the pip packages used for export tooling
- [ ] `specs/tech-stack.md` matches the files that own each pin
- [ ] Any completed roadmap phase is ticked `[x]` and its heading marked `(Complete)`
- [ ] If the release intentionally needs no README change, the reason is written in `CHANGELOG.md`

#### Step 4 — CHANGELOG

- Move `[Unreleased]` to `[vX.Y.Z]` with the date; open a fresh empty `[Unreleased]`.
- Review the **Known Issues** table: close what this release fixes, and leave what it does not with
  its reason intact.
- Keep the house style — prose explaining *why*, plus per-file change tables. This project does not
  generate its changelog from `git log`; a bullet per commit would lose the reasoning.

#### Step 5 — Cut it

Git-flow. Confirm with the user before pushing or tagging — these are outward-facing and hard to
undo.

```bash
git checkout develop && git pull
git checkout -b release/vX.Y.Z
# version bumps + CHANGELOG commit here
git checkout master && git merge --no-ff release/vX.Y.Z
git tag -a vX.Y.Z -m "vX.Y.Z"
git checkout develop && git merge --no-ff master
git branch -d release/vX.Y.Z
```

Push `master`, `develop`, and the tag only once the user has approved.

#### Step 6 — After

- [ ] Verify the tag builds clean from a fresh clone with the default backend
- [ ] `specs/roadmap.md` Status section reflects the new baseline
- [ ] Anything deferred out of this release is in the roadmap Deferred table **with its reason**

### Checklist: gpu-verify

*Runs the manual GPU and alternate-backend verification that CI cannot — build the TensorRT/CUDA/DALI matrix, run the four pre/post combinations against the parity tolerances, compute-sanitizer a long video run, record benchmarks. Trigger when the user says "verify the GPU path", "run the GPU gate", "test TensorRT manually", "parity check".*

**CI runners are all `ubuntu-latest` with no GPU. TensorRT, ExecuTorch, DALI and CUDA paths are
never built or run by CI** ([specs/tech-stack.md](specs/tech-stack.md)). This checklist is
the only thing standing between those paths and an unverified release. It is the roadmap Phase 4
exit gate, and it is also required before any release that touches them.

#### Prerequisites

- NVIDIA GPU with the CUDA Toolkit installed manually (TensorRT implies CUDA 13.x)
- The CUDA Toolkit's nvJPEG (`libnvjpeg-dev-<cuda>`), for the default CUDA preprocessor
- DALI staged once, for the alternative preprocessor: `./scripts/fetch_dali.sh` → `~/dependencies/dali`
- A `.engine` or `.onnx` model, plus a test image and a video of at least 1000 frames
- For DALI, checked-in `.dali` pipelines exist for resolutions **432** and **576** only; anything
  else needs `./scripts/generate_dali_pipelines.sh <res>` with `--gpus all` Docker. The CUDA
  preprocessor runs at any resolution

On rented hardware, `./scripts/run_gate.sh` drives steps 1, 2, 4, 5 and 6 below unattended and
reports the rest as `UNRUN`. Renting, preparing and collecting:
[specs/rented-gpu-runbook.md](specs/rented-gpu-runbook.md).

#### 1. Build the matrix

```bash
# TensorRT + full GPU pipeline (CUDA preprocessing + CUDA postprocessing)
cmake -S . -B build-gpu -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON \
      -DUSE_GPU_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release -DWERROR=ON
cmake --build build-gpu --parallel

# The alternative: DALI preprocessing + CUDA postprocessing
cmake -S . -B build-gpu-dali -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON \
      -DUSE_GPU_PIPELINE=ON -DUSE_DALI=ON -DDALI_ROOT=$HOME/dependencies/dali \
      -DCMAKE_BUILD_TYPE=Release -DWERROR=ON
cmake --build build-gpu-dali --parallel
```

Also confirm the halves build independently — `-DUSE_CUDA_PREPROCESS=ON` alone, `-DUSE_DALI=ON`
alone (no nvcc needed) and `-DUSE_CUDA_POSTPROCESS=ON` alone — that any of them with
`USE_ONNX_RUNTIME=ON` still fails at configure time with a `FATAL_ERROR`, and that
`-DUSE_CUDA_PREPROCESS=ON -DUSE_DALI=ON` fails too. Those guards are architectural commitments,
not niceties.

#### 2. The four combinations

Run every fixture in `tests/data/gpu_parity/` through all four, in each GPU pipeline build:

| Combination | Flags |
|-------------|-------|
| CPU / CPU | *(none)* |
| GPU-pre / CPU-post | `--gpu-preprocess` |
| CPU-pre / GPU-post | `--gpu-postprocess --segmentation` |
| GPU / GPU | `--gpu-preprocess --gpu-postprocess --segmentation` |

`--gpu-postprocess` requires `--segmentation`; there is no GPU postprocess for detection or
keypoint, deliberately ([specs/roadmap.md](specs/roadmap.md) → Deferred).

##### Tolerances — every one is a number, none is negotiable

- [ ] Preprocessed tensor, frame path: CUDA kernel `max |Δ| ≤ 1e-5`; DALI `max |Δ| ≤ 2e-2` (a
      tolerance gate, never equality — DALI resize will not bit-match the CPU bilinear)
- [ ] Preprocessed tensor, JPEG decoded on the GPU (nvJPEG, either preprocessor): `max |Δ| ≤ 1e-1`
      — nvJPEG and stb are different decoders; CUDA PNG fallback `max |Δ| ≤ 1e-5`
- [ ] Detection sets match on class and count, scores within `1e-3`
- [ ] Box centres within 1 px
- [ ] Mask IoU ≥ 0.999
- [ ] Assertions are on the **set** of detections, not the order — score-sort ties are the one
      legitimate ordering difference

#### 3. The dense fixture

- [ ] The dense fixture yields **> 100** above-threshold detections, and both paths agree on the
      count. Natural images yield 10–50 and cannot distinguish a truncating postprocessor from a
      correct one, so a run that skips this fixture has not run the gate.

#### 4. Memory and long-run safety

```bash
compute-sanitizer --tool memcheck ./build-gpu/inference_app <model> <video> <labels> \
    --segmentation --gpu-preprocess --gpu-postprocess
```

- [ ] A **1000-frame** video run completes with no leak and **no `compute-sanitizer` findings**
- [ ] Run it on the default (CUDA preprocessing) build, and on the DALI build when DALI changed
- [ ] DALI builds: particular attention to `daliOutputRelease` ordering — release **after** the TensorRT enqueue.
      Getting it wrong produces intermittent garbage, not a crash, so a single clean short run
      proves nothing ([specs/gpu-pipeline.md](specs/gpu-pipeline.md))

#### 5. Benchmarks

```bash
cmake -S . -B build-gpu-bench -G Ninja -DUSE_ONNX_RUNTIME=OFF -DUSE_TENSORRT=ON \
      -DUSE_GPU_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release -DBENCHMARKS=ON
# add -DUSE_DALI=ON -DDALI_ROOT=$HOME/dependencies/dali to time the DALI preprocessor instead
cmake --build build-gpu-bench --parallel && ./build-gpu-bench/benchmarks
```

- [ ] Four stages timed separately — preprocess, H2D+infer, D2H, postprocess — for a still image
      and a video run, in all four combinations
- [ ] Numbers recorded **including the flat ones**. Expect a large win in segmentation postprocess
      and little or no end-to-end gain from GPU preprocessing on single still images; recording
      that it is flat is the result, not a failure

#### 6. The default path is unchanged

- [ ] The default ONNX Runtime CPU build produces **bit-identical** results to before the change.
      Nothing in the GPU pipeline is allowed to move the default path.
- [ ] `ctest --test-dir build --output-on-failure -R UnitTests` passes on a normal build
- [ ] On a machine with `USE_CUDA_POSTPROCESS=ON` but **no** device, GPU tests report `SKIPPED`,
      not `FAILED`, and the skip is visible in the output

#### 7. Record it

- [ ] `CHANGELOG.md` updated with what was verified, on which hardware, and with which driver,
      CUDA, TensorRT and DALI versions
- [ ] Anything **not** verified is stated plainly. An unrun check is reported as unrun, never
      implied to have passed
