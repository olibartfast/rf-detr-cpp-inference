# Plan

1. No GPU: fix pose score computation in normalized/log space; add synthetic regression cases and real-model upstream score comparison.
2. No GPU: remove hardcoded ONNX single-thread ceiling; wire --onnx-threads (0 auto, positive explicit) through factory options; validate CLI/backend and measure same-model outputs/timing at 1/2/4/auto.
3. No GPU: reproduce and fix LiteRT dependency resolution with existing dependencies, a supported constraint in versions.env and checked requirements/setup. Verify clean install/import and export.
4. GPU required: use user-requested Colab to run the existing TensorRT/CUDA/DALI gate and headless display checks on exact source. Real watched-screen playback still requires a display.
5. No GPU: run strict builds, CPU/Python regression and real-model checks, lint and update docs/changelog only for observed fixes. Record exact remaining blockers, without release publication.
