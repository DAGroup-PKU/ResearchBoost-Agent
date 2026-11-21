#!/bin/bash
# Enhanced environment check for the local Ascend ONNX -> OM pipeline.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAILED=0
WARNINGS=0
MIN_CANN_VERSION="${ASCEND_ONNX_ATC_PIPELINE_MIN_CANN:-8.3.RC2}"
PYTHON_BIN="${ASCEND_ONNX_ATC_PIPELINE_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_status() {
    if [ "$1" -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    elif [ "$1" -eq 1 ]; then
        echo -e "${RED}✗${NC} $2"
        FAILED=$((FAILED + 1))
    else
        echo -e "${YELLOW}⚠${NC} $2"
        WARNINGS=$((WARNINGS + 1))
    fi
}

module_installed() {
    local module="$1"
    "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib.util
spec = importlib.util.find_spec("$module")
raise SystemExit(0 if spec is not None else 1)
PY
}

module_version() {
    local module="$1"
    "$PYTHON_BIN" - <<PY 2>/dev/null
import importlib.util
spec = importlib.util.find_spec("$module")
if spec is None:
    raise SystemExit(1)
module = __import__("$module")
print(getattr(module, "__version__", "unknown"))
PY
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

auto_optimizer_ready() {
    "$PYTHON_BIN" -m auto_optimizer optimize --help >/dev/null 2>&1
}

module_version_equals() {
    local module="$1"
    local expected="$2"
    [ "$(module_version "$module")" = "$expected" ]
}

onnx_version_ok() {
    "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import re
import onnx
parts = [int(x) for x in re.findall(r"\d+", onnx.__version__)]
parts = (parts + [0, 0, 0])[:3]
raise SystemExit(0 if tuple(parts) == (1, 16, 1) else 1)
PY
}

echo "=== Ascend ONNX -> OM Environment Check ==="
echo

echo "=== 1. Python ==="
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    print_status 1 "Python interpreter not found: $PYTHON_BIN"
else
    PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1 | awk '{print $2}')
    echo "Python version: $PYTHON_VERSION"
    case "$PYTHON_VERSION" in
        3.11.*)
            print_status 0 "Python 3.11 matches the currently maintained baseline for CANN >= 8.3.RC2"
            ;;
        3.10.*)
            print_status 2 "Python 3.10 may still be usable for older CANN combinations; verify against the installed CANN version"
            ;;
        3.*)
            print_status 1 "Python $PYTHON_VERSION does not match the currently maintained baseline; verify it against the installed CANN version"
            ;;
        *)
            print_status 1 "Unexpected Python version: $PYTHON_VERSION"
            ;;
    esac
fi
echo

echo "=== 2. CANN / ATC ==="
if CANN_INFO="$(detect_cann 2>/dev/null)"; then
    CANN_PATH="${CANN_INFO%%|*}"
    CANN_REST="${CANN_INFO#*|}"
    CANN_VERSION="${CANN_REST%%|*}"
    CANN_SET_ENV="${CANN_REST#*|}"
    echo "CANN path: $CANN_PATH"
    echo "CANN version: $CANN_VERSION"
    if [ -n "$CANN_SET_ENV" ]; then
        echo "CANN set_env.sh: $CANN_SET_ENV"
    else
        print_status 2 "CANN set_env.sh not found near detected CANN path; atc may still work if the environment is already sourced"
    fi
    if version_ge "$CANN_VERSION" "$MIN_CANN_VERSION"; then
        print_status 0 "CANN version satisfies baseline ${MIN_CANN_VERSION}"
    else
        print_status 1 "CANN version is below ${MIN_CANN_VERSION}"
    fi
else
    print_status 1 "CANN installation not found"
fi

if command -v atc >/dev/null 2>&1; then
    print_status 0 "atc is available: $(command -v atc)"
else
    print_status 2 "atc is not in PATH yet; run: source ${SCRIPT_DIR}/setup_env.sh"
fi
echo

echo "=== 3. Core Python Packages ==="
if module_installed numpy; then
    NUMPY_VERSION=$(module_version numpy)
    echo "numpy version: $NUMPY_VERSION"
    if module_version_equals numpy "1.26.4"; then
        print_status 0 "numpy baseline matches 1.26.4"
    else
        print_status 1 "numpy must be aligned to 1.26.4"
    fi
else
    print_status 1 "numpy is missing"
fi

if module_installed onnx; then
    ONNX_VERSION=$(module_version onnx)
    echo "onnx version: $ONNX_VERSION"
    if onnx_version_ok; then
        print_status 0 "onnx baseline matches 1.16.1"
    else
        print_status 1 "onnx must be aligned to 1.16.1"
    fi
else
    print_status 1 "onnx is missing"
fi
echo

echo "=== 4. Pipeline Helpers ==="
for module in \
    onnxruntime \
    onnxslim \
    pandas \
    decorator \
    psutil \
    attr \
    attrs \
    google.protobuf \
    sympy \
    cloudpickle \
    tornado \
    absl
do
    label="${module%%.*}"
    if module_installed "$module"; then
        print_status 0 "module '$label' is installed"
    else
        print_status 1 "module '$label' is missing"
    fi
done

if module_installed onnxruntime; then
    ORT_VERSION=$(module_version onnxruntime)
    echo "onnxruntime version: $ORT_VERSION"
    if module_version_equals onnxruntime "1.15.1"; then
        print_status 0 "onnxruntime baseline matches 1.15.1"
    else
        print_status 1 "onnxruntime must be aligned to 1.15.1"
    fi
else
    print_status 1 "onnxruntime is missing"
fi

if auto_optimizer_ready; then
    print_status 0 "auto_optimizer runner is available"
else
    print_status 1 "auto_optimizer runner is unavailable"
fi
echo

echo "=== 5. OM Runtime Packages ==="
for module in aclruntime ais_bench; do
    if module_installed "$module"; then
        print_status 0 "module '$module' is installed"
    else
        print_status 1 "module '$module' is missing"
    fi
done
echo

echo "=== 6. Device ==="
if command -v npu-smi >/dev/null 2>&1; then
    print_status 0 "npu-smi is available"
    npu-smi info -l 2>/dev/null | grep -E "(NPU ID|Name)" | head -10 || true
else
    print_status 2 "npu-smi not found"
fi
echo

echo "=== Summary ==="
if [ "$FAILED" -eq 0 ] && [ "$WARNINGS" -eq 0 ]; then
    echo -e "${GREEN}✓ Environment is ready.${NC}"
elif [ "$FAILED" -eq 0 ]; then
    echo -e "${YELLOW}⚠ ${WARNINGS} warning(s) found.${NC}"
else
    echo -e "${RED}✗ ${FAILED} error(s) and ${WARNINGS} warning(s) found.${NC}"
    echo "Recommended follow-up:"
    echo "  source ${SCRIPT_DIR}/setup_env.sh"
    echo "  bash ${SCRIPT_DIR}/install_cann.sh --help"
    echo "  ASCEND_ONNX_ATC_PIPELINE_PYTHON=${PYTHON_BIN} bash ${SCRIPT_DIR}/repair_python_env.sh"
    exit 1
fi

exit 0
