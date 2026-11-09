#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
RUNTIME_DIR=/tmp/dflash-minimal-runtime
mkdir -p "${RUNTIME_DIR}/pip-cache"
if [[ ! -f "${PROJECT_DIR}/runtime_copy_provenance.json" ]] || [[ ! -f "${RUNTIME_DIR}/venv/bin/python" ]]; then
  python "${PROJECT_DIR}/bootstrap_runtime.py"
fi
env HTTPS_PROXY= HTTP_PROXY= ALL_PROXY= NO_PROXY='*' "${RUNTIME_DIR}/venv/bin/python" -m pip install --cache-dir "${RUNTIME_DIR}/pip-cache" --index-url https://mirrors.aliyun.com/pypi/simple/ 'transformers==4.57.3' 'accelerate==1.15.0'
DFLASH_DOWNLOAD_SOURCE=modelscope python "${PROJECT_DIR}/download_models.py"
"${RUNTIME_DIR}/venv/bin/python" -m pip check
