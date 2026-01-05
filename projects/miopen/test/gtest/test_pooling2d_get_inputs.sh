#!/bin/bash
# Script to test pooling2d with TEST_GET_INPUT_TENSOR = 1
# This tests the gtest version against ctest when using get_inputs()

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GTEST_DIR="$SCRIPT_DIR"
COMPARE_SCRIPT="$SCRIPT_DIR/compare_pooling2d_dumps.py"

echo "=========================================="
echo "Testing pooling2d with TEST_GET_INPUT_TENSOR = 1"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "$GTEST_DIR/pooling2d.cpp" ]; then
    echo "Error: pooling2d.cpp not found in $GTEST_DIR"
    exit 1
fi

# Instructions for compilation
echo "To test with TEST_GET_INPUT_TENSOR = 1, you have two options:"
echo ""
echo "OPTION 1: Using CMake option (recommended):"
echo "  1. Configure CMake with the option:"
echo "     cmake -DMIOPEN_TEST_POOLING2D_GET_INPUT_TENSOR=ON <build_dir>"
echo "     OR if already configured:"
echo "     cmake -DMIOPEN_TEST_POOLING2D_GET_INPUT_TENSOR=ON <build_dir>"
echo ""
echo "  2. Build the test:"
echo "     cmake --build <build_dir> --target test_pooling2d"
echo ""
echo "  3. Run the test to generate pooling2d_gtest_configs.txt:"
echo "     cd <build_dir>/test/gtest"
echo "     ./test_pooling2d > /dev/null 2>&1"
echo ""
echo "OPTION 2: Using compiler flag directly:"
echo "  1. Add -DTEST_GET_INPUT_TENSOR=1 to compiler flags in CMakeLists.txt"
echo "     or pass it via: cmake -DCMAKE_CXX_FLAGS=\"-DTEST_GET_INPUT_TENSOR=1\""
echo ""
echo "3. Compare with ctest output:"
echo "   python3 $COMPARE_SCRIPT single pooling2d_ctest_configs.txt pooling2d_gtest_configs.txt"
echo ""
echo "Note: The ctest output (pooling2d_ctest_configs.txt) should be generated"
echo "      by running the ctest version of test_pooling2d (without --all flag)"
echo "      which uses get_inputs() by default."
echo ""

# Check if config files exist
if [ -f "$GTEST_DIR/pooling2d_gtest_configs.txt" ]; then
    echo "Current pooling2d_gtest_configs.txt exists:"
    wc -l "$GTEST_DIR/pooling2d_gtest_configs.txt"
    echo ""
fi

if [ -f "$GTEST_DIR/pooling2d_ctest_configs.txt" ]; then
    echo "Current pooling2d_ctest_configs.txt exists:"
    wc -l "$GTEST_DIR/pooling2d_ctest_configs.txt"
    echo ""
fi

echo "=========================================="
echo "Next steps:"
echo "1. Compile with TEST_GET_INPUT_TENSOR=1"
echo "2. Run the test to generate new config file"
echo "3. Generate ctest reference (if not already done)"
echo "4. Compare using: python3 $COMPARE_SCRIPT single <ctest_file> <gtest_file>"
echo "=========================================="

