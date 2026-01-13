// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <gtest/gtest.h>
#include <half/half.hpp>
#include "pooling2d_common.hpp"

std::vector<Pooling2dTestCase> GetPooling2dWideTestCases()
{
    // Cache results to avoid duplicate generation when called multiple times
    static std::vector<Pooling2dTestCase> cached_test_cases;
    static bool cached = false;
    
    if(cached)
    {
        return cached_test_cases;
    }
    
    std::vector<Pooling2dTestCase> test_cases;

    // Dataset 2: Wide window configurations
    // Input shapes matching ctest behavior with --dataset 2
    // From pooling2d.hpp: get_2d_pooling_input_shapes_wide()
    std::vector<std::vector<int>> dataset2_inputs;
#if TEST_GET_INPUT_TENSOR
    // When TEST_GET_INPUT_TENSOR = 1, use get_inputs() function (matching original ctest behavior)
    // NOTE: Ctest uses ALL shapes from get_inputs(), but dataset_id=2 is determined by
    //       lens/strides/pads selection via generate_multi_data (third element = dataset 2)
    int batch_factor                      = 0; // Default batch factor matching original ctest
    std::set<std::vector<int>> in_dim_set = get_inputs<int>(batch_factor);
    dataset2_inputs.assign(in_dim_set.begin(), in_dim_set.end());
    std::cerr << "DEBUG: TEST_GET_INPUT_TENSOR=1, using " << dataset2_inputs.size() 
              << " input shapes from get_inputs()\n";
#else
    // When TEST_GET_INPUT_TENSOR = 0, use predefined shapes matching ctest exactly
    // From pooling2d.hpp get_2d_pooling_input_shapes_wide():
    dataset2_inputs = {
        {1, 3, 255, 255}, {2, 3, 227, 227}, {1, 7, 127, 127}, {1, 1, 410, 400}};
#endif

    // Lens: {{35, 35}, {100, 100}, {255, 255}, {410, 400}} - wide window kernel sizes
    std::vector<std::vector<int>> dataset2_lens = {{35, 35}, {100, 100}, {255, 255}, {410, 400}};

    // Strides: {{1, 1}} - only stride 1 for wide windows
    std::vector<std::vector<int>> dataset2_strides = {{1, 1}};

    // Pads: {{0, 0}} - no padding for wide windows
    std::vector<std::vector<int>> dataset2_pads = {{0, 0}};

    // Dataset 2 uses only uint32 (matching ctest behavior)
    std::vector<miopenIndexType_t> dataset2_index_types = {miopenIndexUint32};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    // Generate cartesian product for dataset 2
    // This matches the original ctest test_pooling2d behavior with --dataset 2
    // IMPORTANT: Order must match ctest exactly: index_type -> mode -> input_shape -> lens -> strides -> pads -> wsidx
    // This is the order in which test_driver processes test cases (based on add() call order)
    // Filter invalid combinations at generation time instead of skipping at runtime
    for(const auto& index_type : dataset2_index_types)
    {
        for(const auto& mode : modes)
        {
            for(const auto& input_dims : dataset2_inputs)
            {
                AddTestCasesForInput(input_dims,
                                     dataset2_lens,
                                     dataset2_strides,
                                     dataset2_pads,
                                     {index_type}, // Single index_type for this iteration
                                     {mode},       // Single mode for this iteration
                                     wsidx_values,
                                     test_cases,
                                     false,  // skip_wide_check=false for Dataset 2 (wide window)
                                     false,  // apply_index_type_limits=false for Dataset 2 (matching ctest)
                                     true);  // is_wide_dataset=true for Dataset 2 (wide window)
            }
        }
    }

    std::cerr << "\n=== Dataset 2 (Wide Window) Test Case Generation Summary ===\n";
    std::cerr << "Total test cases generated: " << test_cases.size() << "\n";
    std::cerr << "Number of input shapes used: " << dataset2_inputs.size() << "\n";
    std::cerr << "Expected: 14 test cases per data type (14x2=28 total with FP32+FP16)\n";
    std::cerr << "Note: When TEST_GET_INPUT_TENSOR=1, ctest uses all shapes from get_inputs()\n";
    std::cerr << "      but only those that pass kernel size validation (lens <= input+2*pad)\n";

    // Analyze by mode and wsidx to identify the pattern
    std::map<int, std::map<int, int>> mode_wsidx_counts;
    for(const auto& tc : test_cases)
    {
        mode_wsidx_counts[static_cast<int>(tc.mode)][tc.wsidx]++;
    }

    std::cerr << "\nBreakdown by mode and wsidx:\n";
    const char* mode_names[] = {"Max", "Average", "AverageInclusive"};
    for(int mode = 0; mode < 3; mode++)
    {
        int total_for_mode = 0;
        for(int wsidx = 0; wsidx < 2; wsidx++)
        {
            int count = mode_wsidx_counts[mode][wsidx];
            total_for_mode += count;
            std::cerr << "  " << mode_names[mode] << " mode, wsidx=" << wsidx << ": " << count
                      << "\n";
        }
        std::cerr << "  " << mode_names[mode] << " mode TOTAL: " << total_for_mode << "\n";
    }

    std::cerr << "=============================================================\n\n";

    // Log all test configurations to a file for comparison with ctest
    // Format: input_dims[4] lens[2] pads[2] strides[2] index_type mode wsidx
    std::ofstream log_file("pooling2d_wide_gtest_configs.txt");
    if(log_file.is_open())
    {
        log_file << "# Total test cases: " << test_cases.size() << "\n";
        log_file << "# Format: input_dims[4] lens[2] pads[2] strides[2] index_type mode wsidx\n";
        for(const auto& tc : test_cases)
        {
            log_file << tc.input_dims[0] << " " << tc.input_dims[1] << " " << tc.input_dims[2]
                     << " " << tc.input_dims[3] << " ";
            log_file << tc.lens[0] << " " << tc.lens[1] << " ";
            log_file << tc.pads[0] << " " << tc.pads[1] << " ";
            log_file << tc.strides[0] << " " << tc.strides[1] << " ";
            log_file << static_cast<int>(tc.index_type) << " " << static_cast<int>(tc.mode) << " "
                     << tc.wsidx << "\n";
        }
        log_file.close();
    }

    // Cache the results
    cached_test_cases = test_cases;
    cached = true;
    
    return test_cases;
}

// Derived classes for Dataset 2 (wide window pooling)
using GPU_WidePooling2d_FP32 = Pooling2dCommon<float>;
using GPU_WidePooling2d_FP16 = Pooling2dCommon<half_float::half>;

TEST_P(GPU_WidePooling2d_FP32, FloatTest_pooling2d_wide) { this->RunTest(); }

TEST_P(GPU_WidePooling2d_FP16, HalfTest_pooling2d_wide) { this->RunTest(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_WidePooling2d_FP32,
                         testing::ValuesIn(GetPooling2dWideTestCases()),
                         GetPooling2dTestCaseName);

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_WidePooling2d_FP16,
                         testing::ValuesIn(GetPooling2dWideTestCases()),
                         GetPooling2dTestCaseName);
