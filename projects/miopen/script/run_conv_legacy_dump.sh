#!/usr/bin/env bash
set -euo pipefail
 
# Build and run legacy-style convolution test executables, dumping configs.
#
# Usage:
#   ./script/run_conv_legacy_dump.sh [build_dir] [jobs]
#
# Defaults:
#   build_dir: <repo_root>/build
#   jobs:      nproc
#
# Output:
#   <build_dir>/ctest_conv_dumps/ctest_<name>--all.txt
#   <build_dir>/ctest_conv_dumps/ctest_<name>--all--limit1.txt
#
# Notes:
# - Uses legacy invocation style: --verbose --all and --verbose --all --limit 1
# - Supports both old CTests and GTest binaries that still accept legacy driver args.
 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${1:-$(cd "${SRC_DIR}/../.." && pwd)/build}"
JOBS="${2:-$(nproc)}"
OUT_DIR="${BUILD_DIR}/ctest_conv_dumps"
 
# Optional: set to 1 to skip very slow "conv2d_find2 --all".
SKIP_SLOW_FIND2="${SKIP_SLOW_FIND2:-0}"
 
TARGETS=(
    test_conv2d
    test_conv2d_bias
    test_conv2d_find2
    test_find_2_conv
    test_conv3d
    test_conv3d_bias
    test_conv3d_find2
    test_immed_conv2d
    test_immed_conv3d
    test_conv_group
)
 
mkdir -p "${BUILD_DIR}" "${OUT_DIR}"
 
echo "[INFO] Configuring CMake in: ${BUILD_DIR}"
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}"
 
echo "[INFO] Building ${#TARGETS[@]} legacy conv targets (jobs=${JOBS})..."
cmake --build "${BUILD_DIR}" --parallel "${JOBS}" --target "${TARGETS[@]}"
 
find_test_bin() {
    local target="$1"
    local candidates=(
        "${BUILD_DIR}/bin/${target}"
        "${BUILD_DIR}/test/${target}"
        "${BUILD_DIR}/test/gtest/${target}"
        "${BUILD_DIR}/${target}"
    )
 
    for c in "${candidates[@]}"; do
        if [[ -x "${c}" ]]; then
            echo "${c}"
            return 0
        fi
    done
    return 1
}
 
run_one() {
    local target="$1"
    local short_name="${target#test_}"
    local bin_path=""
 
    if ! bin_path="$(find_test_bin "${target}")"; then
        echo "[WARN] Binary not found for ${target}, skipping"
        return 0
    fi
 
    local out_all="${OUT_DIR}/ctest_${short_name}--all.txt"
    local out_limit1="${OUT_DIR}/ctest_${short_name}--all--limit1.txt"
 
    if [[ "${target}" == "test_conv2d_find2" && "${SKIP_SLOW_FIND2}" == "1" ]]; then
        echo "[WARN] Skipping ${target} --all (SKIP_SLOW_FIND2=1)"
    else
        echo "[INFO] Running ${target} --all --verbose"
        {
            echo "# MIOPEN_DUMP_CONFIGS=1 ${bin_path} --verbose --all"
            MIOPEN_DUMP_CONFIGS=1 "${bin_path}" --verbose --all
        } > "${out_all}" 2>&1
    fi
 
    echo "[INFO] Running ${target} --all --limit 1 --verbose"
    {
        echo "# MIOPEN_DUMP_CONFIGS=1 ${bin_path} --verbose --all --limit 1"
        MIOPEN_DUMP_CONFIGS=1 "${bin_path}" --verbose --all --limit 1
    } > "${out_limit1}" 2>&1
}
 
for t in "${TARGETS[@]}"; do
    run_one "${t}"
done
 
echo "[INFO] Done. Legacy dumps are in: ${OUT_DIR}"
 
 
