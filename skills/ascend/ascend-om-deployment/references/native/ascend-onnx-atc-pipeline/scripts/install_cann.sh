#!/bin/bash
# Install or upgrade CANN from local .run packages.
# Default behavior is "complete installation" instead of installing only one
# package, because partial installs tend to leave missing toolchain/runtime
# pieces for ONNX -> OM workflows.

set -euo pipefail

MIN_VERSION="${ASCEND_ONNX_ATC_PIPELINE_MIN_CANN:-8.3.RC2}"
INSTALL_PATH="${ASCEND_ONNX_ATC_PIPELINE_CANN_PATH:-/usr/local/Ascend}"
PYTHON_BIN="${ASCEND_ONNX_ATC_PIPELINE_PYTHON:-python3}"
PACKAGE_PATH=""
PACKAGE_DIR=""
TARGET_VERSION=""
OPS_PRODUCT="${ASCEND_ONNX_ATC_PIPELINE_CANN_OPS_PRODUCT:-}"
INSTALL_MODE="complete"
FORCE_INSTALL=0

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

detect_cann_version() {
    local base
    local version_file
    for base in \
        "${ASCEND_ONNX_ATC_PIPELINE_CANN_HOME:-}" \
        "${ASCEND_TOOLKIT_HOME:-}" \
        "${ASCEND_HOME_PATH:-}" \
        "${INSTALL_PATH}/cann/latest" \
        "${INSTALL_PATH}/cann" \
        "${INSTALL_PATH}/cann-"* \
        "${INSTALL_PATH}/ascend-toolkit/latest" \
        "${INSTALL_PATH}/ascend-toolkit" \
        "${INSTALL_PATH}"
    do
        [ -n "$base" ] || continue
        [ -d "$base" ] || continue
        if version_file="$(find_version_file "$base")"; then
            grep "Version=" "$version_file" | head -1 | cut -d'=' -f2
            return 0
        fi
    done
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

left = normalize(sys.argv[1])
right = normalize(sys.argv[2])
raise SystemExit(0 if left >= right else 1)
PY
}

detect_arch() {
    local arch
    arch="$(uname -m)"
    case "$arch" in
        aarch64|arm64)
            echo "aarch64"
            ;;
        x86_64|amd64)
            echo "x86_64"
            ;;
        *)
            echo "Unsupported architecture: ${arch}" >&2
            exit 1
            ;;
    esac
}

extract_version_from_filename() {
    local name="$1"
    local version
    version="$(basename "$name" | sed -n 's/.*_\([0-9][0-9A-Za-z.]*\)_linux-.*/\1/p' | head -1)"
    if [ -n "$version" ]; then
        echo "$version"
    fi
}

first_existing_file() {
    local candidate
    for candidate in "$@"; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

resolve_ops_package() {
    local dir="$1"
    local version="$2"
    local arch="$3"
    local product_raw="$4"
    local product_lower
    local product_upper
    product_lower="$(printf '%s' "$product_raw" | tr '[:upper:]' '[:lower:]')"
    product_upper="$(printf '%s' "$product_raw" | tr '[:lower:]' '[:upper:]')"

    first_existing_file \
        "$dir/Ascend-cann-${product_raw}-ops_${version}_linux-${arch}.run" \
        "$dir/Ascend-cann-${product_lower}-ops_${version}_linux-${arch}.run" \
        "$dir/Ascend-cann-${product_upper}-ops_${version}_linux-${arch}.run"
}

resolve_legacy_kernels_package() {
    local dir="$1"
    local version="$2"
    local arch="$3"
    local product_raw="$4"
    local product_lower
    local product_upper
    product_lower="$(printf '%s' "$product_raw" | tr '[:upper:]' '[:lower:]')"
    product_upper="$(printf '%s' "$product_raw" | tr '[:lower:]' '[:upper:]')"

    if [ "$product_upper" = "A3" ]; then
        first_existing_file \
            "$dir/Atlas-A3-cann-kernels_${version}_linux-${arch}.run" \
            "$dir/Ascend-cann-kernels-A3_${version}_linux-${arch}.run" \
            "$dir/Ascend-cann-kernels-a3_${version}_linux-${arch}.run"
        return
    fi

    first_existing_file \
        "$dir/Ascend-cann-kernels-${product_raw}_${version}_linux-${arch}.run" \
        "$dir/Ascend-cann-kernels-${product_lower}_${version}_linux-${arch}.run" \
        "$dir/Ascend-cann-kernels-${product_upper}_${version}_linux-${arch}.run"
}

require_product() {
    if [ -z "$OPS_PRODUCT" ]; then
        echo "Complete CANN installation requires --ops-product <product>." >&2
        echo "Examples: 910b, 910, 310p, 310b, A3" >&2
        exit 1
    fi
}

resolve_complete_package_list() {
    local dir="$1"
    local version="$2"
    local arch="$3"
    local toolkit
    local ops_or_kernels
    local nnae
    local nnrt

    toolkit="$(first_existing_file "$dir/Ascend-cann-toolkit_${version}_linux-${arch}.run")" || {
        echo "Missing toolkit package in ${dir} for version ${version} (${arch})." >&2
        return 1
    }

    if version_ge "$version" "8.5.0"; then
        require_product
        ops_or_kernels="$(resolve_ops_package "$dir" "$version" "$arch" "$OPS_PRODUCT")" || {
            echo "Missing ops package in ${dir} for product ${OPS_PRODUCT}, version ${version}, arch ${arch}." >&2
            return 1
        }
        printf '%s\n%s\n' "$toolkit" "$ops_or_kernels"
        return 0
    fi

    require_product
    nnae="$(first_existing_file "$dir/Ascend-cann-nnae_${version}_linux-${arch}.run")" || {
        echo "Missing NNAE package in ${dir} for version ${version} (${arch})." >&2
        return 1
    }
    nnrt="$(first_existing_file "$dir/Ascend-cann-nnrt_${version}_linux-${arch}.run")" || {
        echo "Missing NNRT package in ${dir} for version ${version} (${arch})." >&2
        return 1
    }
    ops_or_kernels="$(resolve_legacy_kernels_package "$dir" "$version" "$arch" "$OPS_PRODUCT")" || {
        echo "Missing kernels package in ${dir} for product ${OPS_PRODUCT}, version ${version}, arch ${arch}." >&2
        return 1
    }
    printf '%s\n%s\n%s\n%s\n' "$toolkit" "$nnae" "$nnrt" "$ops_or_kernels"
}

install_package() {
    local package="$1"
    echo "Installing $(basename "$package")"
    bash "$package" --install --quiet --install-path="${INSTALL_PATH}"
}

usage() {
    echo "Usage:"
    echo "  bash $0 --package /path/to/Ascend-cann-toolkit.run --mode single"
    echo "  bash $0 --package-dir /path/to/cann_pkgs --version 8.3.RC2 --ops-product 910b"
    echo
    echo "Options:"
    echo "  --package PATH         Install a single local CANN .run package"
    echo "  --package-dir DIR      Install a complete CANN package set from a local directory"
    echo "  --version VERSION      Target CANN version, default: ${MIN_VERSION}"
    echo "  --ops-product PRODUCT  Chip/product id for ops or kernels package"
    echo "  --mode MODE            complete (default) or single"
    echo "  --install-path DIR     Install root, default: ${INSTALL_PATH}"
    echo "  --force                Install even if an acceptable CANN already exists"
    echo "  -h, --help             Show this help"
    echo
    echo "Defaults:"
    echo "  - 当前脚本默认执行完整安装，不再只装 toolkit。"
    echo "  - 对于 8.5.x 及以上，完整安装默认是 toolkit + 对应产品 ops。"
    echo "  - 对于 8.3.RC2 及更早的 8.3 拆包体系，完整安装默认是"
    echo "    toolkit + nnae + nnrt + 对应产品 kernels。"
    echo "  - NNAL/ATB/SiP 仍按官方文档视为可选加速库，不纳入默认完整安装。"
    echo
    echo "Notes:"
    echo "  - 仅在本机 CANN 缺失或版本低于目标版本时使用。"
    echo "  - 脚本不负责下载官方安装包，需用户先准备本地 .run 文件。"
    echo "  - 若机器已有满足目标版本的 CANN，默认直接跳过。"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --package)
            PACKAGE_PATH="$2"
            shift 2
            ;;
        --package-dir)
            PACKAGE_DIR="$2"
            shift 2
            ;;
        --version)
            TARGET_VERSION="$2"
            shift 2
            ;;
        --ops-product)
            OPS_PRODUCT="$2"
            shift 2
            ;;
        --mode)
            INSTALL_MODE="$2"
            shift 2
            ;;
        --install-path)
            INSTALL_PATH="$2"
            shift 2
            ;;
        --force)
            FORCE_INSTALL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [ -z "$TARGET_VERSION" ] && [ -n "$PACKAGE_PATH" ]; then
    TARGET_VERSION="$(extract_version_from_filename "$PACKAGE_PATH" || true)"
fi
if [ -z "$TARGET_VERSION" ]; then
    TARGET_VERSION="$MIN_VERSION"
fi

if current_version="$(detect_cann_version 2>/dev/null)"; then
    echo "Detected local CANN: ${current_version}"
    if [ "$FORCE_INSTALL" -eq 0 ] && version_ge "$current_version" "$TARGET_VERSION"; then
        echo "Local CANN already satisfies target ${TARGET_VERSION}; skip installation."
        exit 0
    fi
fi

case "$INSTALL_MODE" in
    single|complete)
        ;;
    *)
        echo "Unsupported mode: ${INSTALL_MODE}" >&2
        exit 1
        ;;
esac

ARCH="$(detect_arch)"

if [ "$INSTALL_MODE" = "single" ]; then
    if [ -z "$PACKAGE_PATH" ]; then
        echo "--mode single requires --package PATH" >&2
        exit 1
    fi
    if [ ! -f "$PACKAGE_PATH" ]; then
        echo "Package not found: $PACKAGE_PATH" >&2
        exit 1
    fi
    install_package "$PACKAGE_PATH"
else
    if [ -z "$PACKAGE_DIR" ]; then
        echo "Complete installation requires --package-dir DIR" >&2
        exit 1
    fi
    if [ ! -d "$PACKAGE_DIR" ]; then
        echo "Package directory not found: $PACKAGE_DIR" >&2
        exit 1
    fi

    mapfile -t PACKAGES_TO_INSTALL < <(resolve_complete_package_list "$PACKAGE_DIR" "$TARGET_VERSION" "$ARCH")
    if [ "${#PACKAGES_TO_INSTALL[@]}" -eq 0 ]; then
        echo "No installable packages resolved from ${PACKAGE_DIR}" >&2
        exit 1
    fi

    echo "Resolved complete CANN package set:"
    printf '  %s\n' "${PACKAGES_TO_INSTALL[@]}"
    for package in "${PACKAGES_TO_INSTALL[@]}"; do
        install_package "$package"
    done
fi

echo "CANN installation command finished."
echo "Next step:"
echo "  source the matching set_env.sh under ${INSTALL_PATH}"
echo "  or run: source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_env.sh"
