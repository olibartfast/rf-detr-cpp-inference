# Running the GPU Gate on a Rented GPU

CI runners have no GPU, so the TensorRT, DALI and CUDA paths are verified by hand via
[gpu-verify](../AGENTS.md#checklist-gpu-verify). This document is the operational half: how to
rent a machine, run [`scripts/run_gate.sh`](../scripts/run_gate.sh) on it unattended, and not leave
it billing.

The checklist says *what* to check and to which tolerances. This says *how to get a box that can check
it*.

---

## What a rented hour actually buys you today

**Running `run_gate.sh` is not passing the gate.** `tests/data/gpu_parity/` does not exist and
`src/main.cpp` has no output-path flag — roadmap Phases 2 and 1 — so the tolerances the gate is
built around (preprocessed tensor `max |Δ| ≤ 2e-2`, scores within `1e-3`, box centres within 1 px,
mask IoU ≥ 0.999) have nothing to compare against. The script reports them `UNRUN`.

| Gate step | On a rented box today |
|-----------|-----------------------|
| 1 — build matrix and configure guards | Fully runnable |
| 2 — four pre/post combinations | Smoke runs only; no tolerance check |
| 3 — dense fixture | `UNRUN`, Phase 2 |
| 4 — `compute-sanitizer` over 1000 frames | Fully runnable |
| 5 — benchmarks | `benchmarks` runs; the per-stage four-combination benchmark is Phase 2 |
| 6 — default CPU path unchanged | Fully runnable, but needs no GPU — do it at home |

**Phase 2 is CPU work you can do locally for free, and it is what makes a rented hour worth
buying.** Doing it first is the difference between closing the gate and paying for four smoke tests.

---

## Choosing an instance

The workload is **build-bound, not GPU-bound**: five CMake configure-and-build cycles dominate the
clock and the GPU is idle for most of them. Pick on cores, not VRAM. Any modern card has ample
memory for RF-DETR at 432.

Priorities, in order:

1. **CPU count.** Five builds on 8 cores is ~30 min; on 128 cores it is ~4.
2. **Provisioning time.** Some providers are ready in 1 minute, others 7. That is 10% of a one-hour
   budget.
3. **Stop/start**, if you plan to spread the work over several sittings rather than one run.
4. **Disk.** `scripts/fetch_dali.sh` pulls a ~25 GB Triton image to extract ~1 GB of DALI. 100 GB
   works; more is comfortable.
5. **VRAM and GPU class.** Last. 16 GB is already plenty.

`run_gate.sh` reads `CUDA_ARCH` from the card's compute capability; set it only to override. For
reference — it is a property of the rented box, not of this project:

| Card | `CUDA_ARCH` |
|------|-------------|
| L4, L40, L40S, RTX 4000/6000 Ada | `89` |
| A4000, A5000, A10, A10G, A6000, A40 | `86` |
| A100 | `80` |
| T4 | `75` |
| RTX PRO 6000 Blackwell (Colab "G4") | `120` |

**Avoid RAM-starved machine families.** A C++ build with nvcc, TensorRT headers under C++20 and
OpenCV routinely peaks at 1–2 GB per translation unit. A `highcpu`-class VM at ~0.9 GB per vCPU
will OOM-kill a parallel build, and cloud VMs generally have no swap. Budget ≥ 2 GB per core.

**Watch the billing model.** Some instances cannot be stopped, only deleted — for those the clock
runs from deploy to delete, so do every bit of preparation before you press Deploy. Where credits
rather than a card fund the account, an exhausted balance may delete the instance *and its disk*;
copy results off as you go rather than at the end.

---

## Part 1 — At home, before renting (free)

### 1. Export a segmentation model at 432

Both details matter: `--gpu-postprocess` requires `--segmentation`, and only **432** and **576**
have checked-in `.dali` pipelines for the DALI build the gate also runs. Anything else needs
`./scripts/generate_dali_pipelines.sh <res>` and a `--gpus all` Docker host. The default CUDA
preprocessor runs at any resolution.

```bash
python3.11 -m venv rfdetr_venv && source rfdetr_venv/bin/activate
pip install -r deploy/requirements.txt   # the pinned rfdetr[onnx]
python deploy/export_segmentation.py --model_type medium --input_size 432
```

See [export.md](../docs/export.md) for the full matrix of export options.

### 2. Prepare a video of at least 1000 frames

A single clean short run proves nothing about stream ordering (and, for DALI, `daliOutputRelease`) — see
[gpu-pipeline.md](gpu-pipeline.md).

```bash
ffmpeg -stream_loop 30 -i clip.mp4 -c copy long.mp4                    # real footage, preferred
ffmpeg -loop 1 -i data/dog.jpg -t 40 -r 25 -pix_fmt yuv420p long.mp4   # fallback: exactly 1000 frames
```

### 3. Run gate step 6 locally

It needs no GPU, so paying for it is waste:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
ctest --test-dir build --output-on-failure -R UnitTests
```

Passing here lets you set `SKIP_DEFAULT_PATH=1` on the rented box. That skips only this CPU build; the GPU build's UnitTests still run there, which is where `test_gpu_postprocess`
actually executes instead of `GTEST_SKIP()`-ing.

### 4. Install your provider's CLI

`run_gate.sh` uses `brev stop` for both its deadline watchdog and its self-stop. Without a CLI on
the instance it falls back to `shutdown -h`, and you must confirm once on the provider console that
a halted VM bills as *stopped* rather than billed-but-off.

---

## Part 2 — Provisioning

Use the provider's setup-script field so the box prepares itself while you do something else.
**TensorRT is not installed here** — `cmake/deps/packages/TensorRT.cmake` downloads the pinned
tarball at configure time. You only need the CUDA toolkit, for `nvcc` and `compute-sanitizer`.

```bash
#!/usr/bin/env bash
set -euxo pipefail

git clone https://github.com/olibartfast/rf-detr-cpp-inference.git ~/rfdetr_inference
cd ~/rfdetr_inference && git checkout develop
source scripts/versions.sh   # CUDA_VERSION

wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y "cuda-toolkit-${CUDA_VERSION//./-}" \
    git ninja-build cmake g++ pkg-config libopencv-dev libgtest-dev tmux rsync

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc

# ~25 GB image pulled to extract ~1 GB of DALI. Drop it afterwards.
./scripts/fetch_dali.sh
(source scripts/versions.sh && docker image rm "${TRITON_IMAGE}") || true

touch ~/SETUP_DONE
```

---

## Part 3 — Running the gate

Wait for `~/SETUP_DONE`, then push the model and video from your machine:

```bash
rsync -av output/rfdetr-seg-medium.onnx long.mp4 <instance>:~/
```

On the box, **inside tmux** — a plain SSH session dies with your laptop lid, and the instance keeps
billing either way:

```bash
cd ~/rfdetr_inference
tmux new -s gate

DEADLINE_HOURS=1 \
CUDA_ARCH=89 \
SKIP_DEFAULT_PATH=1 \
MODEL=~/rfdetr-seg-medium.onnx \
VIDEO=~/long.mp4 \
./scripts/run_gate.sh 2>&1 | tee ~/run.log
```

`Ctrl-b d` detaches; `tmux attach -t gate` resumes from anywhere.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL` | *(unset)* | `.onnx` or `.engine`. Steps 2–5 report `UNRUN` without it |
| `VIDEO` | *(unset)* | ≥ 1000 frames, for `compute-sanitizer` |
| `CUDA_ARCH` | *(the card's compute capability, else `89`)* | `CMAKE_CUDA_ARCHITECTURES` for the rented card |
| `DEADLINE_HOURS` | `6` | Watchdog fires `brev stop` after this, run finished or not |
| `SKIP_DEFAULT_PATH` | `0` | `1` skips step 6's default ONNX Runtime build, which needs no GPU. Step 6's UnitTests on the GPU build still run — `test_gpu_postprocess` needs the device |
| `SKIP_BUILD_MATRIX` | `0` | `1` skips step 1's independent-halves builds and configure guards — card-independent, already compiled by `gpu-compile.yml` — and reports them `UNRUN` |
| `SELF_STOP` | `1` | `0` leaves the instance up after a clean finish |
| `WATCHDOG` | `1` | `0` does not arm the watchdog at all. **Set this when running at home** — without a `brev` CLI the watchdog falls back to `sudo shutdown -h`, which halts your own machine |
| `EXTRA_CMAKE_ARGS` | *(empty)* | Extra `-D` flags for the four TensorRT configures, e.g. `-DTENSORRT_ROOTDIR=<prefix>` on a box that already has TensorRT |
| `REPO`, `DALI_ROOT`, `RESULTS` | derived / `~/dependencies/dali` / `~/gate-results` | Paths |

### Protecting a one-hour budget

`compute-sanitizer` under memcheck runs 10–50× slower than native, so a 1000-frame run is anywhere
from 10 to 40 minutes — the only step that can blow the budget. Start with a ~300-frame clip. A
recorded 300-frame pass beats a 1000-frame run you killed at the deadline, and the summary records
the difference honestly.

---

## Part 4 — Collect, then kill

```bash
rsync -av <instance>:~/gate-results/ ./gate-results/
```

`gate-results/` holds `gate.summary` (the `PASS`/`FAIL`/`UNRUN` table), `gate.log`,
`environment.txt`, per-combination output, `compute-sanitizer.txt`, and `benchmarks.txt`.

The script self-stops on a clean finish and the watchdog fires regardless — but **stopped is not
deleted**, and storage keeps billing on a stopped instance. When you are done with the gate,
delete the environment and confirm on the provider console that it is gone.

Then gate step 7, which the script cannot do for you: put the versions from `environment.txt`
(driver, CUDA, TensorRT, DALI) and the summary into `CHANGELOG.md`, **stating the `UNRUN` items
plainly as unrun**. An unrun check is reported as unrun, never implied to have passed.

---

## Google Colab

Colab is the cheap way to cover cards nobody here owns — T4 (sm_75), L4 (sm_89), A100 (sm_80) and
G4, an RTX PRO 6000 Blackwell (sm_120). The gate itself is unchanged;
[`scripts/colab/gpu_gate.ipynb`](../scripts/colab/gpu_gate.ipynb) drives it and
[`scripts/colab/setup_colab.sh`](../scripts/colab/setup_colab.sh) prepares the runtime.

Open the notebook in Colab (*File → Open notebook → GitHub*, or upload it), choose the GPU under
*Runtime → Change runtime type*, and run the cells in order. Results land on Drive under
`MyDrive/rfdetr-gate/<GPU>-<timestamp>/`, one directory per card.

How Colab differs from a rented box, and what the notebook does about it:

| Colab | Handling |
|-------|----------|
| No Docker | DALI staged with `DALI_SOURCE=pip ./scripts/fetch_dali.sh` — the same wheel tree the Triton image carries, but a PyPI build rather than the NGC one; record that next to the result |
| Preinstalled CUDA is not `CUDA_VERSION` | `setup_colab.sh` installs the pinned toolkit from NVIDIA's apt repo and puts it first on `PATH`/`LD_LIBRARY_PATH` via `/content/rfdetr_gate_env.sh` |
| Driver may predate CUDA 13 (needs R580+) | Installs `cuda-compat` for the pinned series and prepends it. Forward compat works on data-centre cards only, and `compute-sanitizer` may refuse to attach under it — the gate then reports step 4 `UNRUN`, not `FAIL` |
| 2 vCPUs on the standard runtime | Turn High-RAM on; `SKIP_BUILD_MATRIX=1` (notebook default) drops the three card-independent builds |
| Nothing to stop, runtime recycled at will | `WATCHDOG=0 SELF_STOP=0`; results written to Drive as they are produced |
| Idle disconnects | The gate runs in the background; re-run the "follow" cell after reconnecting |
| Colab exports its own `CUDA_VERSION` | `setup_colab.sh` unsets it before sourcing `versions.sh`, which would otherwise let it replace the pin |

`--display` is checked by [`scripts/check_display.sh`](../scripts/check_display.sh) (notebook cells
9–10): it runs the app under Xvfb and fails on `preview disabled`, a missing or wrongly sized
window, a blank window grab, or a `q` that does not end the run cleanly. A plain headless run cannot
prove any of that, because the app silently disables the preview when SDL has no display. It works
on any headless box, not only Colab, and proves the display code, not GPU-accelerated presentation
(Xvfb renders in software).

What a Colab pass adds: per-architecture evidence for steps 1 (full builds only), 2, 4 and 5 —
that the kernels compile and run for that SM and the parity tolerances hold on it. It does not
replace a run with `SKIP_BUILD_MATRIX=0` and Docker-staged DALI on some box, once per release.

Things to watch on specific cards:

- **T4 (sm_75).** Verified 2026-10-08 (CHANGELOG `[Unreleased]`). The TensorRT 11 tarball ships
  `libnvinfer_builder_resource_sm75`, and the Colab driver (580.82.07) runs the CUDA 13.3 build
  under minor-version compatibility without `cuda-compat`. Timings with High-RAM (8 vCPUs):
  setup about 4 min, both full builds about 6 min, `compute-sanitizer` over 1000 frames about 37 min.
- **G4 (sm_120).** Blackwell needs a recent driver; if `setup_colab.sh`'s sanity probe fails there,
  compat cannot help (it is not offered for every Blackwell SKU).
- `CUDA_ARCH` comes from the card, so each run builds for exactly one SM. That is the point — a
  PTX-JIT fallback from another arch would hide arch-specific breakage.
- `setup_colab.sh` assumes Colab's Ubuntu has a matching NVIDIA CUDA apt repo
  (`ubuntu<VERSION_ID>`); a runtime image update can break that.

---

## Known rough edges

- First run against real hardware was 2026-08-26 on an RTX 3060 Laptop; see `CHANGELOG.md` for the
  result and the six fixes it took to get there. It is no longer untried, but it has been tried
  exactly once, on one card, with `-DWERROR=OFF`.
- `src/backends/tensorrt_backend.cpp` does **not** compile under `-DWERROR=ON`, which step 1 uses.
  Until that is fixed (roadmap Phase 1), a full gate run needs `EXTRA_CMAKE_ARGS="-DWERROR=OFF"`,
  and the result must say so.
- `compute-sanitizer` must match the driver. A toolkit older than the driver's CUDA version may
  fail to attach at all; the script now reports that as `UNRUN` rather than `FAIL`, but the
  `daliOutputRelease` ordering check is then simply not done.
- The `GTEST_SKIP()`-without-a-device check is untestable on a GPU box by construction. Verify it on
  a CPU-only host.
- `probe_tensorrt` searches `build-*/_deps/TensorRT-*/`, then `TENSORRT_ROOTDIR`, then the system
  include paths, and reads the version from the `NV_TENSORRT_*` macros. It runs *after* step 1
  because that configure is what downloads TensorRT; before then there is genuinely nothing to find.
