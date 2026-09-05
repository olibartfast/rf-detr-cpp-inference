#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
checker="${repo_root}/scripts/check_dockerfile_parity.sh"
fixture_root="$(mktemp -d)"
trap 'rm -rf "${fixture_root}"' EXIT

write_valid() {
    local file="$1"
    printf '%s\n' \
        '# === shared:first ===' \
        'RUN one' \
        '# === /shared:first ===' \
        '# === shared:second ===' \
        'RUN two' \
        '# === /shared:second ===' > "${fixture_root}/${file}"
}

for file in dockerfile.onnxrt dockerfile.executorch dockerfile.trt; do
    write_valid "${file}"
done

expect_pass() {
    if ! DOCKERFILE_PARITY_ROOT="${fixture_root}" "${checker}" >/dev/null; then
        echo "expected parity check to pass: $1" >&2
        exit 1
    fi
}

expect_fail() {
    if DOCKERFILE_PARITY_ROOT="${fixture_root}" "${checker}" >/dev/null 2>&1; then
        echo "expected parity check to fail: $1" >&2
        exit 1
    fi
}

restore_all() {
    for file in dockerfile.onnxrt dockerfile.executorch dockerfile.trt; do
        write_valid "${file}"
    done
}

expect_pass baseline

sed -i 's/RUN two/RUN drift/' "${fixture_root}/dockerfile.trt"
expect_fail drift
restore_all

sed -i '/shared:second/,+2d' "${fixture_root}/dockerfile.trt"
expect_fail missing-block
restore_all

sed -i '1i# === shared:first ===' "${fixture_root}/dockerfile.trt"
expect_fail nested-block
restore_all

sed -i '1i# === /shared:first ===' "${fixture_root}/dockerfile.trt"
expect_fail unmatched-close
restore_all

sed -i 's|/shared:first|/shared:wrong|' "${fixture_root}/dockerfile.trt"
expect_fail mismatched-close
restore_all

sed -i '/\/shared:second/d' "${fixture_root}/dockerfile.trt"
expect_fail unclosed-block
restore_all

sed -i '/shared:second/,+2p' "${fixture_root}/dockerfile.trt"
expect_fail duplicate-block
restore_all

for file in dockerfile.onnxrt dockerfile.executorch dockerfile.trt; do
    printf 'FROM scratch\n' > "${fixture_root}/${file}"
done
expect_fail empty-marker-set

printf 'Dockerfile parity checker self-test: PASS\n'
