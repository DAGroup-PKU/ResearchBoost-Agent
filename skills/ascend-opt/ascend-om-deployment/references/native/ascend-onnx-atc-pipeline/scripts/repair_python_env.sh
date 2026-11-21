#!/bin/bash
# Align Python dependencies for the local ONNX -> ATC toolchain.

set -euo pipefail

PYTHON_BIN="${ASCEND_ONNX_ATC_PIPELINE_PYTHON:-python3}"
PIP_TIMEOUT="${ASCEND_ONNX_ATC_PIPELINE_PIP_TIMEOUT:-120}"
PIP_RETRIES="${ASCEND_ONNX_ATC_PIPELINE_PIP_RETRIES:-10}"
PIP_RESUME_RETRIES="${ASCEND_ONNX_ATC_PIPELINE_PIP_RESUME_RETRIES:-10}"
MSIT_BIN="${ASCEND_ONNX_ATC_PIPELINE_MSIT_BIN:-}"
OM_WHEEL_DIR="${ASCEND_ONNX_ATC_PIPELINE_OM_WHEEL_DIR:-}"
ACLRUNTIME_WHEEL="${ASCEND_ONNX_ATC_PIPELINE_ACLRUNTIME_WHEEL:-}"
AIS_BENCH_WHEEL="${ASCEND_ONNX_ATC_PIPELINE_AIS_BENCH_WHEEL:-}"
AIS_BENCH_VERSION="${ASCEND_ONNX_ATC_PIPELINE_AIS_BENCH_VERSION:-0.0.2}"
AIS_BENCH_WHEEL_BASE_URL="${ASCEND_ONNX_ATC_PIPELINE_AIS_BENCH_WHEEL_BASE_URL:-https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer}"
AUTO_DOWNLOAD_OM_RUNTIME="${ASCEND_ONNX_ATC_PIPELINE_AUTO_DOWNLOAD_OM_RUNTIME:-1}"
CACHE_BASE="${ASCEND_ONNX_ATC_PIPELINE_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/ascend-onnx-atc-pipeline}"
CONFLICT_PACKAGES_RAW="${ASCEND_ONNX_ATC_PIPELINE_CONFLICT_PACKAGES:-}"
CORE_REQUIREMENTS=(
    "numpy==1.26.4"
    "onnx==1.16.1"
    "onnxruntime==1.15.1"
    onnxslim
    pandas
    decorator
    attrs
    absl-py
    psutil
    protobuf
    sympy
    cloudpickle
    tornado
)

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

module_installed() {
    module_version "$1" >/dev/null 2>&1
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

numpy_version_ok() {
    "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import numpy as np
raise SystemExit(0 if np.__version__ == "1.26.4" else 1)
PY
}

auto_optimizer_ready() {
    "$PYTHON_BIN" -m auto_optimizer optimize --help >/dev/null 2>&1
}

resolve_first_match() {
    local pattern="$1"
    local matches=()
    mapfile -t matches < <(compgen -G "$pattern")
    if [ "${#matches[@]}" -gt 0 ]; then
        printf '%s\n' "${matches[0]}"
        return 0
    fi
    return 1
}

resolve_runtime_wheels() {
    local aclruntime="${ACLRUNTIME_WHEEL}"
    local aisbench="${AIS_BENCH_WHEEL}"

    if [ -z "$aclruntime" ] && [ -n "$OM_WHEEL_DIR" ]; then
        aclruntime="$(resolve_first_match "$OM_WHEEL_DIR"/aclruntime-*.whl || true)"
    fi
    if [ -z "$aisbench" ] && [ -n "$OM_WHEEL_DIR" ]; then
        aisbench="$(resolve_first_match "$OM_WHEEL_DIR"/ais_bench-*.whl || true)"
    fi

    printf '%s|%s\n' "$aclruntime" "$aisbench"
}

module_requirements_ok() {
    module_installed numpy && numpy_version_ok || return 1
    module_installed onnx && onnx_version_ok || return 1
    for module in onnxruntime onnxslim pandas decorator attr attrs google.protobuf sympy psutil absl cloudpickle tornado; do
        module_installed "$module" || return 1
    done
    return 0
}

print_usage() {
    echo "Usage: $0 [--conflict-packages \"pkg_a pkg_b ...\"]"
    echo
    echo "Optional arguments:"
    echo "  --conflict-packages   Explicit conflict package set to jointly install with the core requirements"
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --conflict-packages)
                if [ "$#" -lt 2 ]; then
                    echo "--conflict-packages requires a value" >&2
                    exit 1
                fi
                CONFLICT_PACKAGES_RAW="$2"
                shift 2
                ;;
            --conflict-packages=*)
                CONFLICT_PACKAGES_RAW="${1#*=}"
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown argument: $1" >&2
                print_usage >&2
                exit 1
                ;;
        esac
    done
}

pip_install() {
    local args=(
        install
        --default-timeout "$PIP_TIMEOUT"
        --retries "$PIP_RETRIES"
    )
    if "$PYTHON_BIN" -m pip help install 2>/dev/null | grep -q -- '--resume-retries'; then
        args+=(--resume-retries "$PIP_RESUME_RETRIES")
    fi
    "$PYTHON_BIN" -m pip "${args[@]}" "$@"
}

pip_check_log() {
    local log_path="$1"
    "$PYTHON_BIN" -m pip check 2>&1 | tee "$log_path"
}

collect_conflict_packages_from_log() {
    local log_path="$1"
    "$PYTHON_BIN" - "$log_path" <<'PY'
import pathlib
import re
import sys

log_text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
patterns = [
    re.compile(r"^\s*([A-Za-z0-9_.-]+)\s+.*\s+requires\s+.+\s+but you have\s+", re.IGNORECASE),
    re.compile(r"^([A-Za-z0-9_.-]+)\s+.+\s+requires\s+", re.IGNORECASE),
    re.compile(r"^\s*([A-Za-z0-9_.-]+)\s+.*\s+depends on\s+", re.IGNORECASE),
    re.compile(r"^([A-Za-z0-9_.-]+)\s+has requirement\s+", re.IGNORECASE),
]
seen = set()
for raw_line in log_text.splitlines():
    line = raw_line.strip()
    if not line:
        continue
    for pattern in patterns:
        match = pattern.match(line)
        if match:
            name = match.group(1)
            normalized = name.lower().replace("_", "-")
            if normalized not in seen:
                seen.add(normalized)
                print(name)
            break
PY
}

unique_packages() {
    local package
    declare -A seen=()
    for package in "$@"; do
        [ -n "$package" ] || continue
        local normalized="${package,,}"
        normalized="${normalized//_/-}"
        if [ -n "${seen[$normalized]+x}" ]; then
            continue
        fi
        seen["$normalized"]=1
        printf '%s\n' "$package"
    done
}

filter_packages_in_log() {
    local log_path="$1"
    shift
    "$PYTHON_BIN" - "$log_path" "$@" <<'PY'
import pathlib
import re
import sys

log_text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
candidates = {}
for package in sys.argv[2:]:
    normalized = package.lower().replace("_", "-")
    candidates[normalized] = package

patterns = [
    re.compile(r"^\s*([A-Za-z0-9_.-]+)\s+.*\s+requires\s+.+\s+but you have\s+", re.IGNORECASE),
    re.compile(r"^\s*([A-Za-z0-9_.-]+)\s+.*\s+depends on\s+", re.IGNORECASE),
    re.compile(r"^\s*([A-Za-z0-9_.-]+)\s+has requirement\s+", re.IGNORECASE),
    re.compile(r"^ERROR:\s+Could not find a version that satisfies the requirement\s+([A-Za-z0-9_.-]+)", re.IGNORECASE),
    re.compile(r"^ERROR:\s+No matching distribution found for\s+([A-Za-z0-9_.-]+)", re.IGNORECASE),
]

seen = set()
for raw_line in log_text.splitlines():
    line = raw_line.strip()
    if not line:
        continue
    for pattern in patterns:
        match = pattern.match(line)
        if not match:
            continue
        normalized = match.group(1).lower().replace("_", "-")
        if normalized in candidates and normalized not in seen:
            seen.add(normalized)
            print(candidates[normalized])
        break
PY
}

subtract_package_lists() {
    local package
    declare -A remove=()

    while IFS= read -r package; do
        [ -n "$package" ] || continue
        local normalized="${package,,}"
        normalized="${normalized//_/-}"
        remove["$normalized"]=1
    done

    for package in "$@"; do
        [ -n "$package" ] || continue
        local normalized="${package,,}"
        normalized="${normalized//_/-}"
        if [ -z "${remove[$normalized]+x}" ]; then
            printf '%s\n' "$package"
        fi
    done
}

install_core_requirements() {
    local tmp_dir
    local install_log
    local retry_log
    local pip_check_after_retry_log
    local package
    local -a conflict_packages=()
    local -a next_conflict_packages=()
    local -a unresolved_conflicts=()
    local -a dropped_conflicts=()
    local core_install_status=0
    local core_available=0
    local retry_index=0

    tmp_dir="$(mktemp -d)"
    install_log="${tmp_dir}/pip-install.log"

    set +e
    pip_install "${CORE_REQUIREMENTS[@]}" 2>&1 | tee "$install_log"
    core_install_status=${PIPESTATUS[0]}
    set -e

    if [ "$core_install_status" -eq 0 ]; then
        core_available=1
    fi

    if [ -n "$CONFLICT_PACKAGES_RAW" ]; then
        read -r -a conflict_packages <<< "$CONFLICT_PACKAGES_RAW"
    else
        while IFS= read -r package; do
            [ -n "$package" ] || continue
            conflict_packages+=("$package")
        done < <(collect_conflict_packages_from_log "$install_log")
    fi

    if [ "${#conflict_packages[@]}" -gt 0 ]; then
        mapfile -t conflict_packages < <(unique_packages "${conflict_packages[@]}")
    fi

    if [ "$core_install_status" -eq 0 ] && [ "${#conflict_packages[@]}" -eq 0 ]; then
        rm -rf "$tmp_dir"
        return 0
    fi

    if [ "${#conflict_packages[@]}" -eq 0 ]; then
        echo "Core dependency install failed and no conflict package could be extracted from the install log. Logs kept under: ${tmp_dir}" >&2
        return 1
    fi

    echo "Retrying install with conflict packages: ${conflict_packages[*]}" >&2

    while [ "${#conflict_packages[@]}" -gt 0 ]; do
        retry_index=$((retry_index + 1))
        retry_log="${tmp_dir}/pip-install-retry-${retry_index}.log"
        pip_check_after_retry_log="${tmp_dir}/pip-check-after-retry-${retry_index}.log"

        set +e
        pip_install "${CORE_REQUIREMENTS[@]}" "${conflict_packages[@]}" 2>&1 | tee "$retry_log"
        core_install_status=${PIPESTATUS[0]}
        set -e

        if [ "$core_install_status" -eq 0 ]; then
            core_available=1
        fi

        unresolved_conflicts=()
        while IFS= read -r package; do
            [ -n "$package" ] || continue
            unresolved_conflicts+=("$package")
        done < <(filter_packages_in_log "$retry_log" "${conflict_packages[@]}")

        if [ "$core_install_status" -eq 0 ] && [ "${#unresolved_conflicts[@]}" -eq 0 ]; then
            if pip_check_log "$pip_check_after_retry_log" >/dev/null; then
                rm -rf "$tmp_dir"
                return 0
            fi

            unresolved_conflicts=()
            while IFS= read -r package; do
                [ -n "$package" ] || continue
                unresolved_conflicts+=("$package")
            done < <(filter_packages_in_log "$pip_check_after_retry_log" "${conflict_packages[@]}")

            if [ "${#unresolved_conflicts[@]}" -eq 0 ]; then
                echo "Joint install succeeded, but pip check still reports unresolved conflicts outside the current conflict package set. Logs kept under: ${tmp_dir}" >&2
                return 1
            fi
        fi

        if [ "${#unresolved_conflicts[@]}" -eq 0 ]; then
            if [ "$core_available" -eq 1 ]; then
                echo "Joint install still failed, but no reducible conflict package could be identified. Keeping the current core stack. Logs kept under: ${tmp_dir}" >&2
                return 0
            fi
            echo "Joint install failed and no reducible conflict package could be identified. Logs kept under: ${tmp_dir}" >&2
            return 1
        fi

        mapfile -t unresolved_conflicts < <(unique_packages "${unresolved_conflicts[@]}")
        for package in "${unresolved_conflicts[@]}"; do
            dropped_conflicts+=("$package")
        done

        mapfile -t next_conflict_packages < <(
            printf '%s\n' "${unresolved_conflicts[@]}" | subtract_package_lists "${conflict_packages[@]}"
        )

        if [ "${#next_conflict_packages[@]}" -eq "${#conflict_packages[@]}" ]; then
            if [ "$core_available" -eq 1 ]; then
                echo "Failed to shrink the conflict package set further. Keeping the current core stack. Logs kept under: ${tmp_dir}" >&2
                return 0
            fi
            echo "Failed to shrink the conflict package set further and the core stack is still unavailable. Logs kept under: ${tmp_dir}" >&2
            return 1
        fi

        conflict_packages=("${next_conflict_packages[@]}")
        if [ "${#conflict_packages[@]}" -gt 0 ]; then
            echo "Reducing conflict packages to: ${conflict_packages[*]}" >&2
        fi
    done

    if [ "$core_available" -eq 1 ]; then
        if [ "${#dropped_conflicts[@]}" -gt 0 ]; then
            mapfile -t dropped_conflicts < <(unique_packages "${dropped_conflicts[@]}")
            echo "Stopped retrying after dropping conflict packages: ${dropped_conflicts[*]}" >&2
        fi
        echo "Core requirements are available. Conflict package repair was narrowed down to an empty set. Logs kept under: ${tmp_dir}" >&2
        return 0
    fi

    echo "Core requirements are still unavailable after exhausting the conflict package set. Logs kept under: ${tmp_dir}" >&2
    echo "Logs kept under: ${tmp_dir}" >&2
    return 1
}

python_cache_tag() {
    "$PYTHON_BIN" - <<'PY'
import sys
print(f"py{sys.version_info.major}{sys.version_info.minor}")
PY
}

python_wheel_tag() {
    "$PYTHON_BIN" - <<'PY'
import sys
major = sys.version_info.major
minor = sys.version_info.minor
if major != 3:
    raise SystemExit("Unsupported Python major version for aclruntime wheels")
if minor == 7:
    print("cp37-cp37m")
else:
    print(f"cp{major}{minor}-cp{major}{minor}")
PY
}

platform_wheel_tag() {
    local arch
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64)
            printf '%s\n' "linux_x86_64"
            ;;
        aarch64|arm64)
            printf '%s\n' "linux_aarch64"
            ;;
        *)
            echo "Unsupported architecture for ais_bench/aclruntime wheels: ${arch}" >&2
            return 1
            ;;
    esac
}

download_file_with_wget() {
    local url="$1"
    local output="$2"
    local temp_output="${output}.part"

    rm -f "$temp_output"

    if command -v wget >/dev/null 2>&1; then
        if wget -q --show-progress -O "$temp_output" "$url"; then
            mv -f "$temp_output" "$output"
            return 0
        fi

        if wget --no-check-certificate -q --show-progress -O "$temp_output" "$url"; then
            mv -f "$temp_output" "$output"
            return 0
        fi
    fi

    if command -v curl >/dev/null 2>&1; then
        if curl -fL --retry 3 --retry-delay 2 -o "$temp_output" "$url"; then
            mv -f "$temp_output" "$output"
            return 0
        fi

        if curl -k -fL --retry 3 --retry-delay 2 -o "$temp_output" "$url"; then
            mv -f "$temp_output" "$output"
            return 0
        fi
    fi

    rm -f "$temp_output"
    echo "Failed to download ${url}. Provide local wheels or check network / certificate settings." >&2
    return 1
}

download_runtime_wheels() {
    local version="$1"
    local py_tag="$2"
    local plat_tag="$3"
    local py_cache_tag
    local wheel_dir
    local acl_name="aclruntime-${version}-${py_tag}-${plat_tag}.whl"
    local ais_name="ais_bench-${version}-py3-none-any.whl"
    local acl_url="${AIS_BENCH_WHEEL_BASE_URL}/${version}/ait/${acl_name}"
    local ais_url="${AIS_BENCH_WHEEL_BASE_URL}/${version}/ait/${ais_name}"
    local acl_path
    local ais_path

    py_cache_tag="$(python_cache_tag)"
    wheel_dir="$CACHE_BASE/om_runtime_wheels/${version}/${py_cache_tag}/${plat_tag}"
    acl_path="$wheel_dir/$acl_name"
    ais_path="$wheel_dir/$ais_name"

    mkdir -p "$wheel_dir"

    if [ ! -s "$acl_path" ]; then
        echo "Downloading ${acl_url}" >&2
        download_file_with_wget "$acl_url" "$acl_path" >&2 || return 1
    fi

    if [ ! -s "$ais_path" ]; then
        echo "Downloading ${ais_url}" >&2
        download_file_with_wget "$ais_url" "$ais_path" >&2 || return 1
    fi

    printf '%s|%s\n' "$acl_path" "$ais_path"
}

resolve_runtime_wheels_with_fallback() {
    local wheels
    local runtime_acl
    local runtime_ais
    local py_tag
    local plat_tag

    wheels="$(resolve_runtime_wheels)"
    runtime_acl="${wheels%%|*}"
    runtime_ais="${wheels#*|}"
    if [ -n "$runtime_acl" ] && [ -n "$runtime_ais" ]; then
        printf '%s\n' "$wheels"
        return 0
    fi

    if [ "$AUTO_DOWNLOAD_OM_RUNTIME" != "1" ]; then
        printf '%s\n' "$wheels"
        return 0
    fi

    py_tag="$(python_wheel_tag)" || return 1
    plat_tag="$(platform_wheel_tag)" || return 1

    echo "OM runtime wheels not provided; downloading official wheels for ${py_tag} ${plat_tag}" >&2
    download_runtime_wheels "$AIS_BENCH_VERSION" "$py_tag" "$plat_tag"
}

parse_args "$@"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi
PYTHON_BIN="$(command -v "$PYTHON_BIN")"

if [ -n "$MSIT_BIN" ]; then
    if command -v "$MSIT_BIN" >/dev/null 2>&1; then
        MSIT_BIN="$(command -v "$MSIT_BIN")"
    fi
else
    MSIT_BIN="$(dirname "$PYTHON_BIN")/msit"
    if [ ! -x "$MSIT_BIN" ] && command -v msit >/dev/null 2>&1; then
        MSIT_BIN="$(command -v msit)"
    fi
fi

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    echo "pip is not available for ${PYTHON_BIN}" >&2
    exit 1
fi

need_install=0

if module_installed numpy && numpy_version_ok; then
    echo "numpy ready: $(module_version numpy)"
else
    echo "numpy will be aligned to 1.26.4"
    need_install=1
fi

if module_installed onnx && onnx_version_ok; then
    echo "onnx ready: $(module_version onnx)"
else
    echo "onnx will be aligned to 1.16.1"
    need_install=1
fi

for module in onnxruntime onnxslim pandas decorator psutil attr attrs absl google.protobuf sympy cloudpickle tornado; do
    if module_installed "$module"; then
        echo "module ready: ${module}"
    else
        echo "module missing: ${module}"
        need_install=1
    fi
done

if [ "$need_install" -eq 1 ]; then
    install_core_requirements
fi

if ! module_requirements_ok; then
    echo "Core ONNX/ATC Python dependencies are still inconsistent after repair." >&2
    exit 1
fi

if auto_optimizer_ready; then
    echo "auto_optimizer ready"
else
    echo "auto_optimizer not ready, attempting repair"
    pip_install msit

    if [ ! -x "$MSIT_BIN" ]; then
        echo "msit executable not found." >&2
        echo "PYTHON_BIN: $PYTHON_BIN" >&2
        echo "MSIT_BIN: $MSIT_BIN" >&2
        echo "Hint: set ASCEND_ONNX_ATC_PIPELINE_MSIT_BIN=/path/to/msit" >&2
        exit 1
    fi

    "$MSIT_BIN" install surgeon

    if auto_optimizer_ready; then
        echo "auto_optimizer ready"
    else
        echo "auto_optimizer still unavailable after msit surgeon installation." >&2
        exit 1
    fi
fi

if module_installed aclruntime && module_installed ais_bench; then
    echo "OM runtime packages ready"
    exit 0
fi

RUNTIME_WHEELS="$(resolve_runtime_wheels_with_fallback)"
RUNTIME_ACL="${RUNTIME_WHEELS%%|*}"
RUNTIME_AIS="${RUNTIME_WHEELS#*|}"

if [ -z "$RUNTIME_ACL" ] || [ -z "$RUNTIME_AIS" ]; then
    echo "OM runtime packages are missing and no installable wheels were found." >&2
    echo "Provide ASCEND_ONNX_ATC_PIPELINE_OM_WHEEL_DIR or both" >&2
    echo "ASCEND_ONNX_ATC_PIPELINE_ACLRUNTIME_WHEEL / ASCEND_ONNX_ATC_PIPELINE_AIS_BENCH_WHEEL," >&2
    echo "or allow automatic wheel download from ${AIS_BENCH_WHEEL_BASE_URL}." >&2
    exit 1
fi

pip_install "$RUNTIME_ACL" "$RUNTIME_AIS"

if module_installed aclruntime && module_installed ais_bench; then
    echo "OM runtime packages ready"
else
    echo "OM runtime packages are still unavailable after wheel installation." >&2
    exit 1
fi
