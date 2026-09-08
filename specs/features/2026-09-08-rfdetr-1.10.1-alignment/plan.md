# Plan

1. Inspect the official 1.10.0...1.10.1 diff and classify all five export boundaries (no GPU).
2. Update versions.env, deploy/requirements.txt, live README/docs/spec references and exporter install guidance (no GPU).
3. Run version synchronization, Python tests, default build and unit tests (no GPU).
4. Export an ONNX model with 1.10.1, check its signature, and execute C++ inference (CPU).
5. Record observed results and remaining release gates; retain concise CHANGELOG notes.
