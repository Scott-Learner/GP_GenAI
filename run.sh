#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
RUNTIME_DIR=${DFLASH_RUNTIME:-/tmp/dflash-minimal-runtime}
RESULTS_DIR=${1:-"${PROJECT_DIR}/results-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"}
"${RUNTIME_DIR}/venv/bin/python" "${PROJECT_DIR}/benchmark.py" --output "${RESULTS_DIR}"
"${RUNTIME_DIR}/venv/bin/python" "${PROJECT_DIR}/report.py" "${RESULTS_DIR}"

if "${RUNTIME_DIR}/venv/bin/python" -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["equal_pairs"]<d["total_pairs"] else 1)' "${RESULTS_DIR}/summary.json"; then
  "${RUNTIME_DIR}/venv/bin/python" "${PROJECT_DIR}/diagnose_differences.py" "${RESULTS_DIR}"
  "${RUNTIME_DIR}/venv/bin/python" "${PROJECT_DIR}/report.py" "${RESULTS_DIR}"
fi
