#!/usr/bin/env bash
# Guard byte-identical blocks shared by the backend Dockerfiles.
set -euo pipefail

repo_root="${DOCKERFILE_PARITY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
files=(dockerfile.onnxrt dockerfile.executorch dockerfile.trt)

declare -A blocks_by_file=()
declare -A body_by_key=()
failures=0

report_failure() {
    printf '  %-8s %s\n' "$1" "$2"
    failures=$((failures + 1))
}

parse_file() {
    local file="$1" line name open_name="" body="" blocks=""
    local path="${repo_root}/${file}"
    local -A seen=()

    if [[ ! -f "${path}" ]]; then
        report_failure "MISSING" "${file}"
        return
    fi

    while IFS= read -r line || [[ -n "${line}" ]]; do
        if [[ "${line}" =~ ^#\ ===\ shared:([^[:space:]]+)\ ===$ ]]; then
            name="${BASH_REMATCH[1]}"
            if [[ -n "${open_name}" ]]; then
                report_failure "NESTED" "${file}: shared:${name} opened inside shared:${open_name}"
                continue
            fi
            if [[ -n "${seen[${name}]+x}" ]]; then
                report_failure "DUPLICATE" "${file}: shared:${name}"
            fi
            seen["${name}"]=1
            open_name="${name}"
            body="${line}"$'\n'
        elif [[ "${line}" =~ ^#\ ===\ /shared:([^[:space:]]+)\ ===$ ]]; then
            name="${BASH_REMATCH[1]}"
            if [[ -z "${open_name}" ]]; then
                report_failure "UNMATCHED" "${file}: closing shared:${name}"
                continue
            fi
            if [[ "${name}" != "${open_name}" ]]; then
                report_failure "MISMATCH" "${file}: shared:${open_name} closed by shared:${name}"
                open_name=""
                body=""
                continue
            fi
            body+="${line}"
            body_by_key["${file}:${name}"]="${body}"
            blocks+="${name}"$'\n'
            open_name=""
            body=""
        elif [[ -n "${open_name}" ]]; then
            body+="${line}"$'\n'
        fi
    done < "${path}"

    if [[ -n "${open_name}" ]]; then
        report_failure "UNCLOSED" "${file}: shared:${open_name}"
    fi
    if [[ -z "${blocks}" ]]; then
        report_failure "EMPTY" "${file}: no shared blocks"
    fi
    blocks_by_file["${file}"]="${blocks}"
}

echo "Checking shared blocks across backend Dockerfiles:"
for file in "${files[@]}"; do
    parse_file "${file}"
done

reference="${files[0]}"
reference_names="${blocks_by_file[${reference}]-}"
for file in "${files[@]:1}"; do
    if [[ "${blocks_by_file[${file}]-}" != "${reference_names}" ]]; then
        report_failure "BLOCKS" "${file}: shared-block names/order differ from ${reference}"
    fi
done

while IFS= read -r name; do
    [[ -n "${name}" ]] || continue
    reference_body="${body_by_key[${reference}:${name}]-}"
    for file in "${files[@]:1}"; do
        body="${body_by_key[${file}:${name}]-}"
        if [[ -z "${body}" ]]; then
            report_failure "MISSING" "${file}: shared:${name}"
        elif [[ "${body}" != "${reference_body}" ]]; then
            report_failure "DRIFT" "${file}: shared:${name} differs from ${reference}"
        fi
    done
done <<< "${reference_names}"

if (( failures > 0 )); then
    printf '\nDockerfile shared-block parity failed with %d error(s).\n' "${failures}"
    exit 1
fi

printf '  ok       %d shared blocks agree across all three files\n' \
    "$(printf '%s' "${reference_names}" | sed '/^$/d' | wc -l)"
