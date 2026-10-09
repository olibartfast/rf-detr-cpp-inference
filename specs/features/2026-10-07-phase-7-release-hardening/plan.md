# Plan — Phase 7, release hardening

Roles follow [`specs/delegation-workflow.md`](../../delegation-workflow.md), Claude Code mapping
(D-5):

- **Driver (this session).** Acts as reasoner and planner. It writes the specs and packets,
  judges each diff and owns the docs. It does not edit `src/`, `tests/` or the build files.
- **`implementer`.** One packet per dispatch, in its own git worktree. It runs
  `./scripts/scoreboard.sh` exactly once, at the end, and reports the result whether it
  passed or failed. It does not repair after a failing scoreboard.
- **`reviewer`.** Read-only. It gives the accept/reject verdict on each packet's diff before
  the driver merges it into the phase branch.

A rejected packet goes back as a fresh, corrected packet. The driver does not patch the
worker's output.

Packets run concurrently only when their writable sets are disjoint. P1, P2 and P3 are disjoint
(only P1 touches `CMakeLists.txt`). P4 needs all three merged first. P5 needs P4. P6 needs P5,
plus the user lowering `perf_event_paranoid`.

## Group A — Code hygiene (no GPU)

### 1. P1: CLI unit and `main()` split (R-1, R-2, R-3 for `main.cpp`)

1. Create `src/cli_options.hpp` and `src/cli_options.cpp` in namespace `rfdetr::cli`, with
   `struct CliOptions`, `struct ParseOutcome` and
   `ParseOutcome parse_cli(std::span<const char *const> args)`. Error text is returned in the
   outcome; nothing is printed inside the parser.
2. Register `src/cli_options.cpp` in `RFDETR_SOURCES` (`CMakeLists.txt:242`).
3. Create `tests/unit/test_cli_options.cpp` and register it in `UNIT_TEST_SOURCES`
   (`CMakeLists.txt:426`). It covers every flag, every error message, the range checks, the
   `none` sentinel, unknown-flag ignoring, and the trailing value flag being ignored (Q-1 pins
   today's behaviour).
4. Rewrite `src/main.cpp`:
   - `print_usage` keeps the backend-specific strings.
   - `build_config`, `run_video`, `run_image` and `print_results` become separate steps.
   - `main` returns the same codes.
   - `std::endl` becomes `'\n'`, except where a flush is observable before a crash path. None
     exists today, so all of them are replaced.
   - The nested ternary is flattened.
   - The empty catches go: `parse_cli` uses `std::from_chars`, which does not throw.
     - `std::from_chars` for `float` needs libstdc++ 11+ (GCC 11+). The project floor is
       GCC 12+, so that is satisfied.
     - The `std::stoi` / `std::stof` semantics must be kept exactly: leading whitespace is
       accepted, and a leading `+` is accepted.
     - `from_chars` rejects both, so the parser must strip them first. The tests pin
       `" 5"` and `"+5"` to today's outcome, measured on the baseline build.
     - Hex and `inf`/`nan` handling for floats must match `std::stof` too. The implementer
       measures them on the baseline binary and pins them in tests. Where `from_chars` cannot
       match, it keeps `std::stoi` / `std::stof` inside a non-empty catch instead.

### 2. P2: Orchestrator and helpers (R-3, R-4)

5. In `src/rfdetr_inference.cpp`:
   - Apply `modernize-pass-by-value` at lines 40 and 76.
   - Flatten the nested conditional operators at lines 60 and 65.
   - Drop the `std::move` of the trivially copyable `BoundingBox` at lines 205, 280 and 530.
   - Split `postprocess_keypoint_outputs` (line 312) into private helpers declared in
     `src/rfdetr_inference.hpp`, each under the complexity threshold. Output stays
     bit-identical.
6. `src/rfdetr_inference.hpp:30`: `enum class ModelType : std::uint8_t` (D-4).
7. `src/rfdetr_types.hpp:17` and `src/media.cpp:233`: replace the C array with `std::array`.
8. `src/backends/onnx_runtime_backend.cpp:39,47,53`: replace `std::endl` with `'\n'`.

### 3. P3: FFmpeg reader and writer (R-3)

9. `src/video_reader.cpp`:
   - Lines 80-82 and 222-224: use `std::array` plus `.data()`, or a scoped
     `NOLINT(<check>)` with a reason (D-3).
   - Lines 102-103: initialise in the member initializer list.
10. `src/video_writer.cpp`:
    - Lines 65-67, 229-232 and 251-253: same rule as task 9.
    - Line 117: the empty catch gets a handling comment and a `NOLINT` with its reason, or is
      restructured.
    - Line 173: explicit bool conversion.
    - Line 212: make `encode_and_write` `const` only if it is logically const. Otherwise use
      `NOLINT` with a reason.

### 4. P4: Enforce (R-5)

11. `.clang-tidy`: `WarningsAsErrors: '*'`. Acceptance is the `lint.yml` command reporting zero
    findings, plus the scoreboard.

## Group B — Profiling (no GPU; needs the user for `perf_event_paranoid`)

### 5. P5: Profiling build and CPU benchmarks (perf spec R-12, R-13, R-15, R-16)

12. `CMakeLists.txt`: the `PROFILING` option, taken from `stash@{0}` (after the sanitizer
    block). Default `OFF`. Uses `add_compile_options`, so it reaches `rfdetr_inference_lib`.
13. `tests/benchmark/bench_cpu_pipeline.cpp`: from `stash@{0}^3`, adapted to the post-P2 API.
    Register it in the `benchmarks` target (`CMakeLists.txt:482` block).
14. Confirm `-DBENCHMARKS=ON -DWERROR=ON` builds without DALI. The investigation's V-21 found
    `encode_jpeg` outside its `#ifdef USE_DALI` guard, so check whether that is still true on
    `develop` and fix it only if the build fails.

### 6. P6: Measurement (perf spec R-7…R-11, R-14)

15. Execute the investigation's `protocol.md` on the quiet host. Write
    `specs/features/2026-09-22-performance-memory-investigation/results.md`, with raw artifacts
    under `/tmp/rfdetr-profile-results/`. Writable: `results.md` only.

## Group C — GPU and display (needs this machine's GPU and the user's display)

16. Build the `inference_app` GPU variants (CUDA pipeline and DALI pipeline) from the phase
    branch inside `rfdetr-p6:*`.
    - Both must compile under `-DWERROR=ON`. CI's `gpu-compile.yml` builds only the library,
      so the P1 `main.cpp` refactor is compiled with GPU macros nowhere else.
17. GPU golden run.
    - Run the same four GPU combinations on `data/dog.jpg` with
      `rfdetr-seg-nano-576-ngc25.12.engine` or a rebuilt TensorRT 11 engine, once with the
      baseline commit `cc0d4f9` and once with the phase tip.
    - Outputs must be byte-identical (R-6 on the GPU path).
18. `--display` (R-7).
    - Run video with `--display`, in both GPU builds, with `--gpu-preprocess --gpu-postprocess
      --segmentation`. Also run CPU/CPU in the GPU build, so a failure can be separated from
      the GPU path.
    - The user watches the window, and the result is recorded per Q-2.

## Group D — Close (driver)

19. `AGENTS.md` and `specs/tech-stack.md`:
    - clang-tidy is enforced, so the "pre-commit gate" line now matches CI.
    - The `PROFILING` option.
    - Propagate to `README.md` and `docs/advanced-usage.md` in the same commit (Spec Sync).
20. `CHANGELOG.md` `[Unreleased]`: prose plus a per-file table.
21. Tick Phase 7 in `specs/roadmap.md`, and record the dropped stash/branch in the commit message.
