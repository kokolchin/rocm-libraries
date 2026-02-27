#!/usr/bin/env bash
set -euo pipefail

# Compiles and runs convolution gtests, dumping generated configurations.
#
# Usage:
#   ./script/run_conv_gtests_dump.sh [build_dir]
#
# Defaults:
#   build_dir: /home/kvkol/build
#
# Output:
#   <build_dir>/gtest_conv_dumps/gtest_<name>--full.txt
#   <build_dir>/gtest_conv_dumps/gtest_<name>--smoke.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${1:-$(cd "${SRC_DIR}/../.." && pwd)/build}"
OUT_DIR="${BUILD_DIR}/gtest_conv_dumps"

mkdir -p "${OUT_DIR}"

# Force HIP toolchain defaults unless caller already provided overrides.

export PATH="/opt/rocm/llvm/bin:/opt/rocm/bin:${PATH}"

export CC="${CC:-/opt/rocm/llvm/bin/clang}"

export CXX="${CXX:-/opt/rocm/llvm/bin/clang++}"

export CFLAGS="${CFLAGS:-} -D__HIP_PLATFORM_AMD__"

export CXXFLAGS="${CXXFLAGS:-} -D__HIP_PLATFORM_AMD__"
 
echo "[INFO] Toolchain:"

echo "       CC=${CC}"

echo "       CXX=${CXX}"

echo "       CFLAGS=${CFLAGS}"

echo "       CXXFLAGS=${CXXFLAGS}"
 
echo "[INFO] Cleaning previous CMake cache in: ${BUILD_DIR}"

rm -f "${BUILD_DIR}/CMakeCache.txt"

rm -rf "${BUILD_DIR}/CMakeFiles"
 
echo "[INFO] Configuring CMake in: ${BUILD_DIR}"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \

    -DCMAKE_BUILD_TYPE=Release \

    -DMIOPEN_BACKEND=HIP \

    -DCMAKE_C_COMPILER="${CC}" \

    -DCMAKE_CXX_COMPILER="${CXX}"

GTEST_TARGETS=(
    test_conv2d
    test_conv2d_bias
    test_conv2d_find2
    test_find_2_conv
    test_conv3d
    test_conv3d_bias
    test_conv3d_find2
    test_immed_conv2d
    test_immed_conv3d
)

echo "[INFO] Building ${#GTEST_TARGETS[@]} conv gtest targets..."
cmake --build "${BUILD_DIR}" --target "${GTEST_TARGETS[@]}"

find_gtest_bin() {
    local target="$1"
    local candidates=(
        "${BUILD_DIR}/bin/${target}"
        "${BUILD_DIR}/test/gtest/${target}"
        "${BUILD_DIR}/test/${target}"
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

    if ! bin_path="$(find_gtest_bin "${target}")"; then
        echo "[WARN] Binary not found for ${target}, skipping"
        return 0
    fi

    local out_full="${OUT_DIR}/gtest_${short_name}--full.txt"
    local out_smoke="${OUT_DIR}/gtest_${short_name}--smoke.txt"

    echo "[INFO] Running ${target} (Full suite)"
    {
        echo "# MIOPEN_DUMP_CONFIGS=1 ${bin_path} --gtest_filter=Full.*"
        MIOPEN_DUMP_CONFIGS=1 "${bin_path}" --gtest_color=no --gtest_filter=Full.*
    } > "${out_full}" 2>&1

    echo "[INFO] Running ${target} (Smoke suite)"
    {
        echo "# MIOPEN_DUMP_CONFIGS=1 ${bin_path} --gtest_filter=Smoke.*"
        MIOPEN_DUMP_CONFIGS=1 "${bin_path}" --gtest_color=no --gtest_filter=Smoke.*
    } > "${out_smoke}" 2>&1
}

for t in "${GTEST_TARGETS[@]}"; do
    run_one "${t}"
done

echo "[INFO] Done. Dumps are in: ${OUT_DIR}"
