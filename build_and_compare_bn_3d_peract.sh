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
OLD_COMMIT=${OLD_COMMIT:-6ad849d0033^}  # Commit before conversion

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
    
    # Checkout the commit
    echo "Checking out commit $commit_hash..."
    git checkout "$commit_hash"
    
    # If building old commit, skip gtest version to avoid duplicate target
    if [ "$commit_hash" = "$OLD_COMMIT" ]; then
        echo "Old commit detected - skipping gtest version to avoid duplicate target..."
        GTEST_CMAKELISTS="$SOURCE_DIR/projects/miopen/test/gtest/CMakeLists.txt"
        if [ -f "$GTEST_CMAKELISTS" ]; then
            # Check if bn_3d_peract_test.cpp is already in SKIP_TESTS
            if ! grep -q "bn_3d_peract_test.cpp" "$GTEST_CMAKELISTS" 2>/dev/null; then
                # Add bn_3d_peract_test.cpp to SKIP_TESTS
                # First, try to modify existing SKIP_TESTS line
                if grep -q "set(SKIP_TESTS dumpTensorTest.cpp)" "$GTEST_CMAKELISTS"; then
                    # Modify the existing SKIP_TESTS line
                    sed -i 's/set(SKIP_TESTS dumpTensorTest.cpp)/set(SKIP_TESTS dumpTensorTest.cpp bn_3d_peract_test.cpp)/' "$GTEST_CMAKELISTS"
                elif grep -q "^set(SKIP_TESTS" "$GTEST_CMAKELISTS"; then
                    # Append to existing SKIP_TESTS (any format)
                    sed -i '/^set(SKIP_TESTS/s/)$/ bn_3d_peract_test.cpp)/' "$GTEST_CMAKELISTS"
                else
                    # Add new SKIP_TESTS line after the OPENCL check block
                    sed -i '/endif()/a\set(SKIP_TESTS bn_3d_peract_test.cpp)' "$GTEST_CMAKELISTS"
                fi
                echo "Added bn_3d_peract_test.cpp to SKIP_TESTS in gtest CMakeLists.txt"
            fi
        fi
    fi
    
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
    
    echo "Running $test_name..."
    for i in $(seq 1 $ITERATIONS); do
        echo "  Iteration $i/$ITERATIONS..."
        local start=$(date +%s.%N)
        eval "$test_cmd" > /dev/null 2>&1
        local end=$(date +%s.%N)
        local duration=$(echo "$end - $start" | bc)
        times+=($duration)
        echo "    Time: ${duration}s"
    done
    
    # Calculate average
    local sum=0
    for t in "${times[@]}"; do
        sum=$(echo "$sum + $t" | bc)
    done
    local avg=$(echo "scale=3; $sum / $ITERATIONS" | bc)
    
    echo "  Average time: ${avg}s"
    echo ""
    
    echo "$avg"
}

# Get current commit to restore later
CURRENT_COMMIT=$(git rev-parse HEAD)
echo "Current commit: $CURRENT_COMMIT"
echo "Will restore to this commit after comparison"
echo ""

# Step 1: Build new gtest version (current branch)
echo "=========================================="
echo "Step 1: Building NEW gtest version"
echo "=========================================="
mkdir -p build_new
build_test "$CURRENT_COMMIT" "$BUILD_DIR" "$BUILD_JOBS" "build_new/test_bn_3d_peract_test"

# Step 2: Build old ctest version
echo "=========================================="
echo "Step 2: Building OLD ctest version"
echo "=========================================="
mkdir -p build_old
build_test "$OLD_COMMIT" "$BUILD_DIR" "$BUILD_JOBS" "build_old/test_bn_3d_peract_test"

# Step 3: Restore to original commit
echo "Restoring to original commit: $CURRENT_COMMIT"
git checkout "$CURRENT_COMMIT"

# Step 4: Run timing comparison
echo "=========================================="
echo "Step 3: Running Timing Comparison"
echo "=========================================="

OLD_BINARY="build_old/test_bn_3d_peract_test"
NEW_BINARY="build_new/test_bn_3d_peract_test"

if [ ! -f "$OLD_BINARY" ]; then
    echo "Error: Old ctest binary not found: $OLD_BINARY"
    exit 1
fi

if [ ! -f "$NEW_BINARY" ]; then
    echo "Error: New gtest binary not found: $NEW_BINARY"
    exit 1
fi

echo "Old ctest binary: $OLD_BINARY"
echo "New gtest binary: $NEW_BINARY"
echo ""

# Run old ctest
OLD_TIME=$(run_and_time "./$OLD_BINARY" "Old CTest")

# Run new gtest (with filter)
NEW_TIME=$(run_and_time "./$NEW_BINARY --gtest_filter=Smoke/GPU_Bn3dPerAct_FP32.*" "New GTest")

# Print summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Old CTest average: ${OLD_TIME}s"
echo "New GTest average: ${NEW_TIME}s"

DIFF=$(echo "$NEW_TIME - $OLD_TIME" | bc)
PERCENT=$(echo "scale=2; ($NEW_TIME / $OLD_TIME) * 100" | bc)
echo "Difference: ${DIFF}s (${PERCENT}% of old time)"

if (( $(echo "$NEW_TIME > $OLD_TIME * 1.1" | bc -l) )); then
    echo "WARNING: New gtest is more than 10% slower than old ctest!"
elif (( $(echo "$NEW_TIME < $OLD_TIME * 0.9" | bc -l) )); then
    echo "SUCCESS: New gtest is more than 10% faster than old ctest!"
else
    echo "OK: New gtest timing is similar to old ctest (within 10%)"
fi

echo ""
echo "To exclude from TheRock, add to SKIP_TESTS in CMakeLists.txt:"
echo "  list(APPEND SKIP_TESTS bn_3d_peract_test.cpp)"
