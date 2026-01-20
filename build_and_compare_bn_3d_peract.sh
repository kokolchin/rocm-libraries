#!/bin/bash
# Script to build both old ctest and new gtest versions, then compare their timing
# This script is designed to run from /data/rocm-libraries on AMD machine
#
# Usage: ./build_and_compare_bn_3d_peract.sh [iterations] [build_jobs]
#
# Example:
#   ./build_and_compare_bn_3d_peract.sh 5 12
#   # Runs 5 timing iterations, uses 12 parallel build jobs

set -e

ITERATIONS=${1:-5}
BUILD_JOBS=${2:-12}
BUILD_DIR=${BUILD_DIR:-/data/build}
SOURCE_DIR=${SOURCE_DIR:-/data/rocm-libraries}
OLD_COMMIT=${OLD_COMMIT:-develop}  # Use develop branch for old ctest version

echo "=========================================="
echo "BN 3D PerAct Build and Timing Comparison"
echo "=========================================="
echo "Iterations: $ITERATIONS"
echo "Build jobs: $BUILD_JOBS"
echo "Build directory: $BUILD_DIR"
echo "Source directory: $SOURCE_DIR"
echo "Old commit: $OLD_COMMIT"
echo ""

# Check if we're in the right directory
if [ ! -f "projects/miopen/test/gtest/bn_3d_peract_test.cpp" ]; then
    echo "Error: Must run from rocm-libraries root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Function to build the test
build_test() {
    local commit_hash="$1"
    local build_dir="$2"
    local jobs="$3"
    local binary_dest="$4"
    
    echo "=========================================="
    echo "Building commit: $commit_hash"
    echo "=========================================="
    
    # Check if binary already exists
    if [ -f "$binary_dest" ] && [ -x "$binary_dest" ]; then
        echo "Binary already exists at $binary_dest, skipping build..."
        echo "Binary will be used for timing comparison."
        echo ""
        return 0
    fi
    
    # Checkout the commit/branch
    echo "Checking out $commit_hash..."
    git checkout "$commit_hash"
    
    # Clean and prepare build directory
    echo "Cleaning build directory: $build_dir..."
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    
    # Configure
    echo "Configuring with cmake..."
    cd "$build_dir"
    cmake "$SOURCE_DIR/projects/miopen"
    
    # Build the target
    echo "Building test_bn_3d_peract_test with -j $jobs..."
    cmake --build . --target test_bn_3d_peract_test -j "$jobs"
    
    # Find the binary
    # When building from projects/miopen, binaries are in bin/
    local binary_path=""
    if [ -f "bin/test_bn_3d_peract_test" ]; then
        binary_path="bin/test_bn_3d_peract_test"
    elif [ -f "test/gtest/test_bn_3d_peract_test" ]; then
        binary_path="test/gtest/test_bn_3d_peract_test"
    elif [ -f "test/test_bn_3d_peract_test" ]; then
        binary_path="test/test_bn_3d_peract_test"
    elif [ -f "test_bn_3d_peract_test" ]; then
        binary_path="test_bn_3d_peract_test"
    else
        echo "Error: Binary not found after build!"
        echo "Searched in: bin/test_bn_3d_peract_test, test/gtest/test_bn_3d_peract_test, test/test_bn_3d_peract_test, test_bn_3d_peract_test"
        exit 1
    fi
    
    echo "Found binary: $build_dir/$binary_path"
    
    # Create destination directory
    mkdir -p "$(dirname "$binary_dest")"
    
    # Copy binary to destination
    echo "Copying binary to $binary_dest..."
    cp "$build_dir/$binary_path" "$binary_dest"
    chmod +x "$binary_dest"
    
    echo "Build complete. Binary saved to: $binary_dest"
    echo ""
    
    cd "$SOURCE_DIR"
}

# Function to run and time a test
run_and_time() {
    local test_cmd="$1"
    local test_name="$2"
    local times=()
    
    echo "Running $test_name..." >&2
    for i in $(seq 1 $ITERATIONS); do
        echo "  Iteration $i/$ITERATIONS..." >&2
        local start=$(date +%s.%N)
        # Check if command exists before running
        if ! eval "$test_cmd" > /dev/null 2>&1; then
            echo "    Error: Command failed with exit code $?: $test_cmd" >&2
            # Try running again without silencing to show the error
            eval "$test_cmd"
            exit 1
        fi
        local end=$(date +%s.%N)
        local duration=$(awk -v end="$end" -v start="$start" 'BEGIN {print end - start}')
        times+=($duration)
        echo "    Time: ${duration}s" >&2
    done
    
    # Calculate average
    local sum=0
    for t in "${times[@]}"; do
        sum=$(awk -v sum="$sum" -v t="$t" 'BEGIN {print sum + t}')
    done
    local avg=$(awk -v sum="$sum" -v iter="$ITERATIONS" 'BEGIN {printf "%.3f", sum / iter}')
    
    echo "  Average time: ${avg}s" >&2
    echo "" >&2
    
    echo "$avg"
}

# Get current commit to restore later
CURRENT_COMMIT=$(git rev-parse HEAD)
SOURCE_DIR=$(pwd) # Ensure SOURCE_DIR is the absolute path to the root
echo "Current commit: $CURRENT_COMMIT"
echo "Source directory: $SOURCE_DIR"
echo "Will restore to this commit after comparison"
echo ""

# Step 1: Build new gtest version (current branch)
echo "=========================================="
echo "Step 1: Building NEW gtest version"
echo "=========================================="
mkdir -p "$SOURCE_DIR/build_new"
build_test "$CURRENT_COMMIT" "$BUILD_DIR" "$BUILD_JOBS" "$SOURCE_DIR/build_new/test_bn_3d_peract_test"

# Step 2: Build old ctest version
echo "=========================================="
echo "Step 2: Building OLD ctest version"
echo "=========================================="
mkdir -p "$SOURCE_DIR/build_old"
# The original CTest target name is test_bn_3d_peract (not test_bn_3d_peract_test)
build_test "$OLD_COMMIT" "$BUILD_DIR" "$BUILD_JOBS" "$SOURCE_DIR/build_old/test_bn_3d_peract"

# Step 3: Restore to original commit
echo "Restoring to original commit: $CURRENT_COMMIT"
git checkout "$CURRENT_COMMIT"

# Step 4: Run timing comparison
echo "=========================================="
echo "Step 3: Running Timing Comparison"
echo "=========================================="

OLD_BINARY="$SOURCE_DIR/build_old/test_bn_3d_peract"
NEW_BINARY="$SOURCE_DIR/build_new/test_bn_3d_peract_test"

if [ ! -x "$OLD_BINARY" ]; then
    echo "Error: Old ctest binary not executable or not found: $OLD_BINARY"
    ls -l "$OLD_BINARY"
    exit 1
fi

if [ ! -x "$NEW_BINARY" ]; then
    echo "Error: New gtest binary not executable or not found: $NEW_BINARY"
    ls -l "$NEW_BINARY"
    exit 1
fi

echo "Old ctest binary: $OLD_BINARY"
echo "New gtest binary: $NEW_BINARY"
echo ""

DATA_TYPES=("float" "half" "double" "int8" "bfloat16")
GTEST_FIXTURES=("FP32" "FP16" "FP64" "INT8" "BF16")

for i in "${!DATA_TYPES[@]}"; do
    DT=${DATA_TYPES[$i]}
    FX=${GTEST_FIXTURES[$i]}
    
    echo "------------------------------------------"
    echo "Testing Data Type: $DT"
    echo "------------------------------------------"
    
    # Run old ctest
    OLD_TIME=$(run_and_time "$OLD_BINARY --all --$DT" "Old CTest ($DT)")
    
    # Run new gtest
    NEW_TIME=$(run_and_time "$NEW_BINARY --gtest_filter=Smoke/GPU_Bn3dPerAct_$FX.*" "New GTest ($DT)")
    
    # Print summary for this type
    echo "Summary for $DT:"
    echo "  Old CTest average: ${OLD_TIME}s"
    echo "  New GTest average: ${NEW_TIME}s"
    
    DIFF=$(awk -v new="$NEW_TIME" -v old="$OLD_TIME" 'BEGIN {print new - old}')
    PERCENT=$(awk -v new="$NEW_TIME" -v old="$OLD_TIME" 'BEGIN {if (old > 0) printf "%.2f", (new / old) * 100; else print "0.00"}')
    echo "  Difference: ${DIFF}s (${PERCENT}% of old time)"
    echo ""
done

echo "=========================================="
echo "Final Summary"
echo "=========================================="
# (Optional: Add total sum comparison here if needed)

echo ""
echo "To exclude from TheRock, add to SKIP_TESTS in CMakeLists.txt:"
echo "  list(APPEND SKIP_TESTS bn_3d_peract_test.cpp)"
