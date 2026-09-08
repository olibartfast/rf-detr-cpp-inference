# Plan

1. Fetch and recover the latest OpenCode session and the older preparation records.
2. Reproduce the catalog failure on current develop, apply the recovered bounded
   repair, register eight CMake regressions in CTest and the regular CI job, and correct
   its README contract.
3. Check the recovered GPU documentation against current CPU/CUDA source.
4. Run native strict build, real-model tests, pin/shared-Docker checks and all
   twelve supported Docker builds; exercise the NVIDIA images on local hardware.
5. Record evidence and resolve the Phase 4 release-gate decision before release.
6. Reconcile project versions, cut CHANGELOG, validate a fresh default build,
   integrate to master, tag the tested production commit and backmerge to develop.
7. Push and verify exact remote commits and CI. Do not claim unrun gates passed.
