#!/usr/bin/env bash
# Stage a native CPU SDK; Python is used only to acquire the official wheel.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/scripts/versions.sh"
prefix="${1:-${HOME}/dependencies/litert}"
python_bin="${PYTHON:-python3}"
staging="$(mktemp -d)"
trap 'rm -rf "${staging}"' EXIT
"${python_bin}" -m venv "${staging}/venv"
"${staging}/venv/bin/python" -m pip install --disable-pip-version-check --no-deps "ai-edge-litert==${LITERT_VERSION}"
"${staging}/venv/bin/python" - "${prefix}" "${LITERT_VERSION}" "${staging}" <<'PY'
import importlib.util
import pathlib
import shutil
import sys
import urllib.request
import zipfile

prefix, version, staging = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
# The wheel contains the standalone C runtime, not a Python extension dependency.
package = pathlib.Path(importlib.util.find_spec('ai_edge_litert').origin).parent
libraries = list(package.rglob('libLiteRt.so'))
if len(libraries) != 1:
    raise SystemExit('Expected one Linux libLiteRt.so in the official wheel; supply an SDK prefix on other platforms')
url = f'https://github.com/google-ai-edge/LiteRT/releases/download/v{version}/litert_cc_sdk.zip'
archive = staging / 'sdk.zip'
urllib.request.urlretrieve(url, archive)
with zipfile.ZipFile(archive) as sdk:
    sdk.extractall(staging / 'sdk')
headers = staging / 'sdk/litert_cc_sdk/litert/c'
(prefix / 'include/litert').mkdir(parents=True, exist_ok=True)
(prefix / 'lib').mkdir(parents=True, exist_ok=True)
shutil.copytree(headers, prefix / 'include/litert/c', dirs_exist_ok=True)
config_dir = prefix / 'include/litert/build_common'
config_dir.mkdir(parents=True, exist_ok=True)
config = (staging / 'sdk/litert_cc_sdk/litert/build_common/build_config.h.in').read_text()
config = config.replace('#cmakedefine01 LITERT_BUILD_CONFIG_DISABLE_GPU', '#define LITERT_BUILD_CONFIG_DISABLE_GPU 0')
config = config.replace('#cmakedefine01 LITERT_BUILD_CONFIG_DISABLE_NPU', '#define LITERT_BUILD_CONFIG_DISABLE_NPU 0')
(config_dir / 'build_config.h').write_text(config)
shutil.copy2(libraries[0], prefix / 'lib/libLiteRt.so')
(prefix / 'VERSION').write_text(version + '\n')
print(f'LiteRT native SDK staged at {prefix.resolve()}')
PY
