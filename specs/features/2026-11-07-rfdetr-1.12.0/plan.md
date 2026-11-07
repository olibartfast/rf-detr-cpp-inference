# Plan

1. No GPU: update versions.env and all enforced restatements; review README/export guidance and removed ONNX dependencies.
2. No GPU: repair opt-in LiteRT keypoint adapter around early upstream rejection; preserve graph metadata, upstream sampling and hook restoration; add Python regression tests. No CMake target changes.
3. No GPU: run version sync, adapter tests, strict default build/unit tests and static checks.
4. Device-dependent: re-export detection/segmentation/pose and execute applicable ONNX, ExecuTorch, LiteRT and TensorRT parity. Record exact unavailable cases.
5. No GPU: record validation and upstream delta in CHANGELOG and this spec. Commit/push only after mandatory lint gates; leave unverified gates open.
