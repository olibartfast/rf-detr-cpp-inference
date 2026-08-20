#!/usr/bin/env bash
# Single pass/fail gate for delegated work (see docs/opencode-workflow.md).
# Prints one verdict line and exits 0 only if every stage passed. Workers run
# this exactly once, at the end, and report the result without repairing it.
#
# Override the build directory with BUILD_DIR=<dir> (default: build). Mirrors
# the `default` CMake preset plus the -DWERROR=ON that CI uses.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
BUILD_DIR="${BUILD_DIR:-build}"

fail() { echo "SCOREBOARD: FAIL ($1)"; exit 1; }

echo "== format =="
CF=$(command -v clang-format-18 || command -v clang-format || true)
[ -n "$CF" ] || fail "clang-format not found; install clang-format-18"
files=$(find src tests -name '*.cpp' -o -name '*.hpp')
# shellcheck disable=SC2086
"$CF" --dry-run --Werror $files || fail "clang-format"

echo "== configure =="
cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DUSE_ONNX_RUNTIME=ON -DWERROR=ON || fail "cmake configure"

echo "== build =="
cmake --build "$BUILD_DIR" --parallel || fail "build"

echo "== test =="
ctest --test-dir "$BUILD_DIR" --output-on-failure || fail "ctest"

echo "SCOREBOARD: PASS"
