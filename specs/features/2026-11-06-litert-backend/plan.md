# Implementation plan

All groups run without a GPU. Subtasks are numbered continuously.

1. Build and dependency contract
   1. Add LITERT_VERSION to versions.env; register cmake/deps/packages/LiteRT.cmake in the existing glob catalog.
   2. Extend exclusive backend selection, dependency resolution, sources and links in CMakeLists.txt (existing sections at lines 109, 229, 271, 332); extend the factory and CLI backend description/keypoint rejection in src/main.cpp and src/cli_options.cpp.
   3. Add scripts/fetch_litert.sh to stage matching native headers/library in an install prefix.
2. Backend
   4. Add src/backends/litert_backend.{hpp,cpp}: RAII model/environment/options/buffers, contract validation, fixed-shape synchronous CPU inference, checked output copies and reinitialization.
   5. Add shared contract helpers src/backends/litert_contract.{hpp,cpp} and meaningful contract and lifecycle unit tests and optional real-model integration tests; register in CMakeLists.txt’s unit/integration source groups.
3. Export and verification
   6. Add the LiteRT CPU CI build job to .github/workflows/ci.yml and deploy/export_litert.py using the pinned upstream direct export API, with detection/segmentation switches.
   7. Export representative models, compare native results against the Python interpreter on identical normalized tensors, and run CLI end to end.
4. Documentation
   8. Update README.md, docs/building.md, docs/export.md, docs/advanced-usage.md, mission, tech-stack, AGENTS, roadmap and open specs affected by the constitution extension.
   9. Record checks in validation.md and add a concise CHANGELOG entry. Leave the phase unticked until merged and verified.

## User-requested extension: pose and a single three-task showcase (no GPU)

10. Verify direct keypoint export with the installed LiteRT converter and fixed input resolution.
11. Extend deploy/export_litert.py, contract tests and CLI support only after successful export.
12. Compare every native keypoint output element with Python LiteRT and inspect skeleton rendering.
13. Run pose on the same video, preserve the existing detection/segmentation outputs, and render
    one YouTube-ready video. The final user correction restricts it to detection and segmentation,
    with truthful runtime/source captions; retain pose only as an experimental code path.
14. Update backend docs and the validation record to reflect exactly what was verified.
