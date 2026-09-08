# Plan

1. Export the official keypoint model with rfdetr 1.10.1 and inspect output shapes (CPU).
2. Add CLI parsing and configuration/tensor-stride validation (CPU).
3. Test active-first, legacy, invalid counts and excessive per-class counts using MockBackend; existing unit target requires no CMake registration (CPU).
4. Build, run unit tests and validate real keypoint inference; document verified flags and any remaining limitations.
