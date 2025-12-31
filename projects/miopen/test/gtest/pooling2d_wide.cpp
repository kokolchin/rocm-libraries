// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <gtest/gtest.h>
#include <half/half.hpp>
#include "pooling2d_common.hpp"

std::vector<Pooling2dTestCase> GetPooling2dWideTestCases()
{
    std::vector<Pooling2dTestCase> test_cases;
    IndexTypeCounters counters;

    // Dataset 2: Wide window configurations
    // Input: {{1, 3, 255, 255}, {2, 3, 227, 227}, {1, 7, 127, 127}, {1, 1, 410, 400}}
    std::vector<std::vector<int>> dataset2_inputs = {
        {1, 3, 255, 255}, {2, 3, 227, 227}, {1, 7, 127, 127}, {1, 1, 410, 400}};

    // Lens: {{35, 35}, {100, 100}, {255, 255}, {410, 400}} - wide window kernel sizes
    std::vector<std::vector<int>> dataset2_lens = {{35, 35}, {100, 100}, {255, 255}, {410, 400}};

    // Strides: {{1, 1}} - only stride 1 for wide windows
    std::vector<std::vector<int>> dataset2_strides = {{1, 1}};

    // Pads: {{0, 0}} - no padding for wide windows
    std::vector<std::vector<int>> dataset2_pads = {{0, 0}};

    std::vector<miopenIndexType_t> dataset2_index_types = {
        miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    // Generate cartesian product for dataset 2
    // This matches the original ctest test_pooling2d behavior with --dataset 2
    // Filter invalid combinations at generation time instead of skipping at runtime
    for(const auto& input_dims : dataset2_inputs)
    {
        AddTestCasesForInput(input_dims,
                             dataset2_lens,
                             dataset2_strides,
                             dataset2_pads,
                             dataset2_index_types,
                             modes,
                             wsidx_values,
                             counters,
                             test_cases,
                             false,  // skip_wide_check=false for Dataset 2 (wide window)
                             false); // apply_index_type_limits=false for Dataset 2 (matching ctest)
    }

    std::cerr << "\n=== Dataset 2 (Wide Window) Test Case Generation Summary ===\n";
    std::cerr << "Total test cases generated: " << test_cases.size() << "\n";
    std::cerr << "Expected ctest count: 33\n";

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
