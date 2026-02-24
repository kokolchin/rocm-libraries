#!/usr/bin/env bash
set -euo pipefail

# Compiles and runs convolution ctests, dumping tested configurations.
#
# Usage:
#   ./script/run_conv_ctests_dump.sh [build_dir]
#
# Defaults:
#   build_dir: /home/kvkol/build
#
# Output:
#   <build_dir>/ctest_conv_dumps/ctest_<name>--all.txt
#   <build_dir>/ctest_conv_dumps/ctest_<name>--all--limit1.txt

SRC_DIR="/home/kvkol/rocm-libraries/projects/miopen"
BUILD_DIR="${1:-/home/kvkol/build}"
TEST_SRC_DIR="${SRC_DIR}/test"
OUT_DIR="${BUILD_DIR}/ctest_conv_dumps"

mkdir -p "${OUT_DIR}"

echo "[INFO] Configuring CMake in: ${BUILD_DIR}"
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}"

mapfile -t CONV_TARGETS < <(
    for f in "${TEST_SRC_DIR}"/*.cpp; do
        b="$(basename "${f}" .cpp)"
        # Ctests in test/ whose basename contains "conv"
        if [[ "${b}" == *conv* ]]; then
            echo "test_${b}"
        fi
    done | sort -u
)

if [[ "${#CONV_TARGETS[@]}" -eq 0 ]]; then
    echo "[ERROR] No conv ctest targets discovered in ${TEST_SRC_DIR}" >&2
    exit 1
fi

echo "[INFO] Building ${#CONV_TARGETS[@]} conv ctest targets..."
cmake --build "${BUILD_DIR}" --target "${CONV_TARGETS[@]}"

run_one() {
    local target="$1"
    local short_name="${target#test_}"
    local bin_path=""

    if [[ -x "${BUILD_DIR}/test/${target}" ]]; then
        bin_path="${BUILD_DIR}/test/${target}"
    elif [[ -x "${BUILD_DIR}/${target}" ]]; then
        bin_path="${BUILD_DIR}/${target}"
    else
        echo "[WARN] Binary not found for ${target}, skipping"
        return 0
    fi

    local out_all="${OUT_DIR}/ctest_${short_name}--all.txt"
    local out_all_limit1="${OUT_DIR}/ctest_${short_name}--all--limit1.txt"

    echo "[INFO] Running ${target} --all"
    {
        echo "# ${bin_path} --all"
        "${bin_path}" --all
    } > "${out_all}" 2>&1

    echo "[INFO] Running ${target} --all --limit 1"
    {
        echo "# ${bin_path} --all --limit 1"
        "${bin_path}" --all --limit 1
    } > "${out_all_limit1}" 2>&1
}

for t in "${CONV_TARGETS[@]}"; do
    run_one "${t}"
done

echo "[INFO] Done. Dumps are in: ${OUT_DIR}"
