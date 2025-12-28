// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include <half/half.hpp>
#include "pooling2d_common.hpp"

// Temporary: Enable configuration logging for comparison with ctest
// Set to 1 to generate a file with all test configurations
// This should be reverted after reviewer verification
#define ENABLE_CONFIG_LOGGING 1

using namespace pooling2d_gtest;

std::vector<Pooling2dTestCase> GetPooling2dAsymmetricTestCases()
{
    std::vector<Pooling2dTestCase> test_cases;
    IndexTypeCounters counters;

    // Dataset 1: Asymmetric configurations
    // Input: {{1, 4, 4, 4}} - minimal input for asymmetric testing
    std::vector<std::vector<int>> dataset1_inputs = {{1, 4, 4, 4}};

    // Lens: {{2, 2}, {1, 2}, {2, 1}} - asymmetric kernel sizes
    std::vector<std::vector<int>> dataset1_lens = {{2, 2}, {1, 2}, {2, 1}};

    // Strides: {{1, 1}, {2, 1}, {1, 2}, {2, 2}} - asymmetric strides
    std::vector<std::vector<int>> dataset1_strides = {{1, 1}, {2, 1}, {1, 2}, {2, 2}};

    // Pads: controlled by WORKAROUND_ISSUE_1670 (matching original ctest behavior)
#if WORKAROUND_ISSUE_1670
    std::vector<std::vector<int>> dataset1_pads = {{0, 0}};
#else
    std::vector<std::vector<int>> dataset1_pads = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
#endif

    std::vector<miopenIndexType_t> dataset1_index_types = {
        miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    // Generate cartesian product for dataset 1
    // This matches the original ctest test_pooling2d behavior with --dataset 1
    // Filter invalid combinations at generation time instead of skipping at runtime
    for(const auto& input_dims : dataset1_inputs)
    {
        AddTestCasesForInput(input_dims,
                             dataset1_lens,
                             dataset1_strides,
                             dataset1_pads,
                             dataset1_index_types,
                             modes,
                             wsidx_values,
                             counters,
                             test_cases,
                             true,  // skip_wide_check=true for Dataset 1 (asymmetric)
                             false); // apply_index_type_limits=false for Dataset 1 (matching ctest)
    }

#if ENABLE_CONFIG_LOGGING
    // Temporary: Log all test configurations to a file for comparison with ctest
    // Format: input_dims[4] lens[2] pads[2] strides[2] index_type mode wsidx
    std::ofstream log_file("pooling2d_asymmetric_gtest_configs.txt");
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
#endif

    return test_cases;
}

// Derived classes for Dataset 1 (asymmetric pooling)
using GPU_AsymPooling2d_FP32 = Pooling2dCommon<float>;
using GPU_AsymPooling2d_FP16 = Pooling2dCommon<half_float::half>;

TEST_P(GPU_AsymPooling2d_FP32, FloatTest_pooling2d_asymmetric) { RunTest(); }

TEST_P(GPU_AsymPooling2d_FP16, HalfTest_pooling2d_asymmetric) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_AsymPooling2d_FP32,
                         testing::ValuesIn(GetPooling2dAsymmetricTestCases()),
                         GetPooling2dTestCaseName);

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_AsymPooling2d_FP16,
                         testing::ValuesIn(GetPooling2dAsymmetricTestCases()),
                         GetPooling2dTestCaseName);
