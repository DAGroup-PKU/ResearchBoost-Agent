#!/bin/bash
# Source CANN environment and repair the local Python toolchain when needed.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MIN_CANN_VERSION="${ASCEND_ONNX_ATC_PIPELINE_MIN_CANN:-8.3.RC2}"
PYTHON_BIN="${ASCEND_ONNX_ATC_PIPELINE_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_version_file() {
    local base="$1"
    local file
    for file in \
        "$base/version.cfg" \
        "$base/version.info" \
        "$base/compiler/version.info" \
        "$base/toolkit/version.info"
    do
        if [ -f "$file" ]; then
            echo "$file"
            return 0
        fi
    done
    return 1
}

find_set_env_script() {
    local base="$1"
    local file
    for file in \
        "$base/set_env.sh" \
        "$base/ascend-toolkit/set_env.sh" \
        "$base/latest/set_env.sh" \
        "$base/toolkit/set_env.sh"
    do
        if [ -f "$file" ]; then
            echo "$file"
            return 0
        fi
    done
    return 1
}

detect_cann() {
    local base
    local version_file
    local version
    local set_env_file
    for base in \
        "${ASCEND_ONNX_ATC_PIPELINE_CANN_HOME:-}" \
        "${ASCEND_TOOLKIT_HOME:-}" \
        "${ASCEND_HOME_PATH:-}" \
        "/usr/local/Ascend/cann/latest" \
        "/usr/local/Ascend/cann" \
        "/usr/local/Ascend/cann-"* \
        "/usr/local/Ascend/ascend-toolkit/latest" \
        "/usr/local/Ascend/ascend-toolkit" \
        "/usr/local/Ascend"
    do
        [ -n "$base" ] || continue
        [ -d "$base" ] || continue
        if version_file="$(find_version_file "$base")"; then
            version="$(grep 'Version=' "$version_file" | head -1 | cut -d'=' -f2)"
            set_env_file="$(find_set_env_script "$base" || true)"
            echo "$base|$version|$set_env_file"
            return 0
        fi
    done

    if command -v atc >/dev/null 2>&1; then
        local atc_bin
        local atc_dir
        atc_bin="$(readlink -f "$(command -v atc)")"
        atc_dir="$(dirname "$(dirname "$atc_bin")")"
        if version_file="$(find_version_file "$atc_dir")"; then
            version="$(grep 'Version=' "$version_file" | head -1 | cut -d'=' -f2)"
            set_env_file="$(find_set_env_script "$atc_dir" || true)"
            echo "$atc_dir|$version|$set_env_file"
            return 0
        fi
    fi
    return 1
}

version_ge() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import re
import sys

def normalize(value: str):
    value = value.strip()
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\.?RC(\d+))?", value, re.I)
    if not m:
        return (-1, -1, -1, -1)
    major = int(m.group(1))
    minor = int(m.group(2))
    patch = int(m.group(3) or 0)
    rc = m.group(4)
    if rc is None:
        rc_rank = 999
    else:
        rc_rank = int(rc)
        patch = -1
    return (major, minor, patch, rc_rank)

raise SystemExit(0 if normalize(sys.argv[1]) >= normalize(sys.argv[2]) else 1)
PY
}

check_npu_device() {
    echo -e "${BLUE}[NPU Device Check]${NC}"
    if command -v npu-smi >/dev/null 2>&1; then
        npu-smi info -l 2>/dev/null | grep -E "(NPU ID|Name)" | head -10 || true
    else
        echo -e "${YELLOW}npu-smi not found${NC}"
    fi
}

setup_environment() {
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo -e "${RED}Python interpreter not found: ${PYTHON_BIN}${NC}"
        return 1
    fi

    if ! CANN_INFO="$(detect_cann 2>/dev/null)"; then
        echo -e "${RED}CANN installation not found.${NC}"
        echo "Run:"
        echo "  bash ${SCRIPT_DIR}/install_cann.sh --help"
        return 1
    fi

    CANN_PATH="${CANN_INFO%%|*}"
    CANN_REST="${CANN_INFO#*|}"
    CANN_VERSION="${CANN_REST%%|*}"
    CANN_SET_ENV="${CANN_REST#*|}"

    echo -e "${GREEN}Detected CANN:${NC} ${CANN_PATH}"
    echo "Version: ${CANN_VERSION}"
    if version_ge "$CANN_VERSION" "$MIN_CANN_VERSION"; then
        echo -e "${GREEN}CANN version satisfies baseline ${MIN_CANN_VERSION}${NC}"
    else
        echo -e "${YELLOW}CANN version is below baseline ${MIN_CANN_VERSION}; continue only if you know this environment is intentional.${NC}"
    fi

    if [ -z "$CANN_SET_ENV" ] || [ ! -f "$CANN_SET_ENV" ]; then
        echo -e "${RED}Missing set_env.sh near detected CANN path: ${CANN_PATH}${NC}"
        echo "If CANN uses a custom layout, set ASCEND_ONNX_ATC_PIPELINE_CANN_HOME to the directory that contains set_env.sh."
        return 1
    fi

    echo -e "${BLUE}Sourcing ${CANN_SET_ENV}${NC}"
    # shellcheck disable=SC1090
    source "${CANN_SET_ENV}"

    if command -v atc >/dev/null 2>&1; then
        echo -e "${GREEN}atc ready: $(command -v atc)${NC}"
    else
        echo -e "${RED}atc is still unavailable after sourcing CANN.${NC}"
        return 1
    fi

    echo -e "${BLUE}Repairing Python toolchain with ${PYTHON_BIN}${NC}"
    ASCEND_ONNX_ATC_PIPELINE_PYTHON="$PYTHON_BIN" bash "${SCRIPT_DIR}/repair_python_env.sh"

    echo -e "${BLUE}Re-checking environment readiness${NC}"
    if ASCEND_ONNX_ATC_PIPELINE_PYTHON="$PYTHON_BIN" bash "${SCRIPT_DIR}/check_env_enhanced.sh"; then
        echo -e "${GREEN}Environment re-check passed.${NC}"
    else
        echo -e "${RED}Environment re-check failed after repair. Stop here and fix the environment before ONNX/ATC.${NC}"
        return 1
    fi

    check_npu_device
}

print_usage() {
    echo "Usage:"
    echo "  source $0"
    echo "  $0"
    echo
    echo "This script:"
    echo "  1. Reuses an existing CANN installation when available"
    echo "  2. Sources the matching set_env.sh"
    echo "  3. Repairs Python dependencies for the ONNX -> OM pipeline"
    echo "  4. Prints basic NPU device information"
    echo
    echo "Optional environment variables for OM runtime wheels:"
    echo "  ASCEND_ONNX_ATC_PIPELINE_OM_WHEEL_DIR=/path/to/wheels"
    echo "  ASCEND_ONNX_ATC_PIPELINE_ACLRUNTIME_WHEEL=/path/to/aclruntime-*.whl"
    echo "  ASCEND_ONNX_ATC_PIPELINE_AIS_BENCH_WHEEL=/path/to/ais_bench-*.whl"
    echo "If no wheels are provided, repair_python_env.sh can auto-download them from:"
    echo "  ASCEND_ONNX_ATC_PIPELINE_AIS_BENCH_WHEEL_BASE_URL=https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
        print_usage
        exit 0
    fi
    echo -e "${YELLOW}Running directly. Environment changes will not persist in the current shell.${NC}"
    setup_environment
else
    setup_environment
fi
