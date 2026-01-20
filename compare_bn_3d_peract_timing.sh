#!/bin/bash
# Script to compare timing between old ctest and new gtest for bn_3d_peract_test
#
# Usage: ./compare_bn_3d_peract_timing.sh [iterations] [old_ctest_binary] [new_gtest_binary]
#
# Workflow:
#   1. Build old ctest version (in a separate build directory to avoid conflicts):
#      git checkout <commit-before-conversion>  # e.g., 6ad849d0033^
#      mkdir build_old && cd build_old
#      cmake .. && cmake --build . --target test_bn_3d_peract_test -j 12
#      # Binary will be at: build_old/test_bn_3d_peract_test
#
#   2. Build new gtest version:
#      git checkout feature/gtest-bn-3d-peract-timing
#      mkdir build_new && cd build_new
#      cmake .. && cmake --build . --target test_bn_3d_peract_test -j 12
#      # Binary will be at: build_new/test_bn_3d_peract_test
#
#   3. Run comparison:
#      ./compare_bn_3d_peract_timing.sh 5 build_old/test_bn_3d_peract_test build_new/test_bn_3d_peract_test
#
# Examples:
#   # Use default paths (searches in build_old and build_new directories)
#   ./compare_bn_3d_peract_timing.sh 5
#
#   # Compare only new gtest (if old ctest not available)
#   ./compare_bn_3d_peract_timing.sh 5 "" ""
#
#   # Specify custom binary paths (absolute or relative)
#   ./compare_bn_3d_peract_timing.sh 5 /path/to/old/binary /path/to/new/binary
#
#   # Use custom build directories
#   BUILD_DIR_OLD=my_old_build BUILD_DIR_NEW=my_new_build ./compare_bn_3d_peract_timing.sh 5

set -e

ITERATIONS=${1:-5}
OLD_CTEST_BINARY_ARG=${2:-""}
NEW_GTEST_BINARY_ARG=${3:-""}
BUILD_DIR_OLD=${BUILD_DIR_OLD:-build_old}
BUILD_DIR_NEW=${BUILD_DIR_NEW:-build_new}

echo "=========================================="
echo "BN 3D PerAct Timing Comparison"
echo "=========================================="
echo "Iterations: $ITERATIONS"
echo ""

# Function to run and time a test
run_and_time() {
    local test_cmd="$1"
    local test_name="$2"
    local times=()
    
    echo "Running $test_name..."
    for i in $(seq 1 $ITERATIONS); do
        echo "  Iteration $i/$ITERATIONS..."
        local start=$(date +%s.%N)
        eval "$test_cmd" > /dev/null 2>&1
        local end=$(date +%s.%N)
        local duration=$(awk "BEGIN {print $end - $start}")
        times+=($duration)
        echo "    Time: ${duration}s"
    done
    
    # Calculate average
    local sum=0
    for t in "${times[@]}"; do
        sum=$(awk "BEGIN {print $sum + $t}")
    done
    local avg=$(awk "BEGIN {printf \"%.3f\", $sum / $ITERATIONS}")
    
    echo "  Average time: ${avg}s"
    echo ""
    
    echo "$avg"
}

# Check if we're in the right directory
if [ ! -f "projects/miopen/test/gtest/bn_3d_peract_test.cpp" ]; then
    echo "Error: Must run from rocm-libraries root directory"
    exit 1
fi

# Determine binary paths
OLD_CTEST_BINARY=""
NEW_GTEST_BINARY=""

# If custom paths provided, use them
if [ -n "$OLD_CTEST_BINARY_ARG" ]; then
    if [ -f "$OLD_CTEST_BINARY_ARG" ]; then
        OLD_CTEST_BINARY="$OLD_CTEST_BINARY_ARG"
        echo "Using provided old ctest binary: $OLD_CTEST_BINARY"
    else
        echo "Error: Provided old ctest binary not found: $OLD_CTEST_BINARY_ARG"
        exit 1
    fi
elif [ -d "$BUILD_DIR_OLD" ]; then
    # Check if old ctest binary exists in build_old directory
    if [ -f "$BUILD_DIR_OLD/projects/miopen/test/test_bn_3d_peract_test" ]; then
        OLD_CTEST_BINARY="$BUILD_DIR_OLD/projects/miopen/test/test_bn_3d_peract_test"
        echo "Found old ctest binary: $OLD_CTEST_BINARY"
    elif [ -f "$BUILD_DIR_OLD/test_bn_3d_peract_test" ]; then
        OLD_CTEST_BINARY="$BUILD_DIR_OLD/test_bn_3d_peract_test"
        echo "Found old ctest binary: $OLD_CTEST_BINARY"
    else
        echo "Warning: Old ctest binary not found in $BUILD_DIR_OLD. Skipping old ctest timing."
        echo "To test old ctest, you can:"
        echo "  1. Build it in $BUILD_DIR_OLD: mkdir $BUILD_DIR_OLD && cd $BUILD_DIR_OLD && cmake .. && cmake --build . --target test_bn_3d_peract_test -j 12"
        echo "  2. Provide the path as second argument: $0 $ITERATIONS /path/to/old/binary"
    fi
else
    echo "Warning: Build directory '$BUILD_DIR_OLD' not found. Skipping old ctest timing."
    echo "To test old ctest, build it in $BUILD_DIR_OLD or provide the path as second argument."
fi

# If custom path provided for new gtest, use it
if [ -n "$NEW_GTEST_BINARY_ARG" ]; then
    if [ -f "$NEW_GTEST_BINARY_ARG" ]; then
        NEW_GTEST_BINARY="$NEW_GTEST_BINARY_ARG"
        echo "Using provided new gtest binary: $NEW_GTEST_BINARY"
    else
        echo "Error: Provided new gtest binary not found: $NEW_GTEST_BINARY_ARG"
        exit 1
    fi
elif [ -d "$BUILD_DIR_NEW" ]; then
    # Check if new gtest binary exists in build_new directory
    if [ -f "$BUILD_DIR_NEW/projects/miopen/test/gtest/test_bn_3d_peract_test" ]; then
        NEW_GTEST_BINARY="$BUILD_DIR_NEW/projects/miopen/test/gtest/test_bn_3d_peract_test"
    elif [ -f "$BUILD_DIR_NEW/test_bn_3d_peract_test" ]; then
        NEW_GTEST_BINARY="$BUILD_DIR_NEW/test_bn_3d_peract_test"
    fi
fi

if [ -z "$NEW_GTEST_BINARY" ] || [ ! -f "$NEW_GTEST_BINARY" ]; then
    echo "Error: New gtest binary not found"
    echo "Please build the project first:"
    echo "  mkdir $BUILD_DIR_NEW && cd $BUILD_DIR_NEW && cmake .. && cmake --build . --target test_bn_3d_peract_test -j 12"
    echo "Or provide the path as third argument: $0 $ITERATIONS \"\" /path/to/new/binary"
    exit 1
fi

if [ -n "$OLD_CTEST_BINARY" ]; then
    echo "Found old ctest binary: $OLD_CTEST_BINARY"
fi
echo "Found new gtest binary: $NEW_GTEST_BINARY"
echo ""

# Run timing comparisons
# Convert relative paths to absolute if needed, and ensure we run from the right directory
if [ -n "$OLD_CTEST_BINARY" ] && [ -f "$OLD_CTEST_BINARY" ]; then
    # Convert to absolute path if relative
    if [[ "$OLD_CTEST_BINARY" != /* ]]; then
        OLD_CTEST_BINARY="$(pwd)/$OLD_CTEST_BINARY"
    fi
    # Get directory of binary for running
    OLD_CTEST_DIR=$(dirname "$OLD_CTEST_BINARY")
    OLD_CTEST_NAME=$(basename "$OLD_CTEST_BINARY")
    OLD_TIME=$(cd "$OLD_CTEST_DIR" && run_and_time "./$OLD_CTEST_NAME" "Old CTest")
else
    OLD_TIME="N/A"
fi

# For gtest, add filter to run only the relevant tests
# Convert to absolute path if relative
if [[ "$NEW_GTEST_BINARY" != /* ]]; then
    NEW_GTEST_BINARY="$(pwd)/$NEW_GTEST_BINARY"
fi
# Get directory of binary for running
NEW_GTEST_DIR=$(dirname "$NEW_GTEST_BINARY")
NEW_GTEST_NAME=$(basename "$NEW_GTEST_BINARY")
NEW_TIME=$(cd "$NEW_GTEST_DIR" && run_and_time "./$NEW_GTEST_NAME --gtest_filter=Smoke/GPU_Bn3dPerAct_FP32.*" "New GTest")

# Print summary
echo "=========================================="
echo "Summary"
echo "=========================================="
if [ "$OLD_TIME" != "N/A" ]; then
    echo "Old CTest average: ${OLD_TIME}s"
fi
echo "New GTest average: ${NEW_TIME}s"

if [ "$OLD_TIME" != "N/A" ]; then
    DIFF=$(awk "BEGIN {print $NEW_TIME - $OLD_TIME}")
    PERCENT=$(awk "BEGIN {printf \"%.2f\", ($NEW_TIME / $OLD_TIME) * 100}")
    echo "Difference: ${DIFF}s (${PERCENT}% of old time)"
    
    if awk "BEGIN {exit !($NEW_TIME > $OLD_TIME * 1.1)}"; then
        echo "WARNING: New gtest is more than 10% slower than old ctest!"
    elif awk "BEGIN {exit !($NEW_TIME < $OLD_TIME * 0.9)}"; then
        echo "SUCCESS: New gtest is more than 10% faster than old ctest!"
    else
        echo "OK: New gtest timing is similar to old ctest (within 10%)"
    fi
fi

echo ""
echo "To exclude from TheRock, add to SKIP_TESTS in CMakeLists.txt:"
echo "  list(APPEND SKIP_TESTS bn_3d_peract_test.cpp)"
