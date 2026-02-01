// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <fstream>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <half/half.hpp>
#include "../network_data.hpp"
#include "pooling2d_common.hpp"

// Configuration define matching the original ctest behavior
// These can be overridden at compile time via -D flags
// TEST_GET_INPUT_TENSOR: When 0, uses all 18 predefined input shapes (matching ctest with --all).
//                        When 1, uses get_inputs() function to generate input shapes from
//                        network_data.
#ifndef TEST_GET_INPUT_TENSOR
#define TEST_GET_INPUT_TENSOR 0
#endif

namespace {

std::vector<pooling2d_gtest::PoolingTestCase> GetPooling2dTestCases()
{
    static std::vector<pooling2d_gtest::PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<pooling2d_gtest::PoolingTestCase> test_cases;

    // Dataset 0: Default dataset (various tensor sizes)
    std::vector<std::vector<int>> dataset0_inputs; 
#if TEST_GET_INPUT_TENSOR
    // When TEST_GET_INPUT_TENSOR = 1, use get_inputs() function (matching original ctest behavior)
    int batch_factor                      = 0; // Default batch factor matching original ctest
    std::set<std::vector<int>> in_dim_set = get_inputs<int>(batch_factor);
    dataset0_inputs.assign(in_dim_set.begin(), in_dim_set.end());
#else
    // When TEST_GET_INPUT_TENSOR = 0, use predefined shapes
    // Use all 18 input shapes to match ctest with --all (limit_set=2, limit_multiplier=9 -> 2*9=18)
    // This matches the maximum number of test cases when running: test_pooling2d --all
    dataset0_inputs = {{1, 19, 1024, 2048},  // Shape 1
                       {10, 3, 32, 32},      // Shape 2
                       {5, 32, 8, 8},        // Shape 3
                       {2, 1024, 12, 12},    // Shape 4
                       {4, 3, 231, 231},     // Shape 5
                       {8, 3, 227, 227},     // Shape 6
                       {1, 384, 13, 13},     // Shape 7
                       {1, 96, 27, 27},      // Shape 8
                       {2, 160, 7, 7},       // Shape 9
                       {1, 192, 256, 512},   // Shape 10
                       {2, 192, 28, 28},     // Shape 11
                       {1, 832, 64, 128},    // Shape 12
                       {1, 256, 56, 56},     // Shape 13
                       {4, 3, 224, 224},     // Shape 14
                       {2, 64, 112, 112},    // Shape 15
                       {2, 608, 4, 4},       // Shape 16
                       {1, 2048, 11, 11},    // Shape 17
                       {1, 16, 4096, 4096}}; // Shape 18
#endif
    std::vector<std::vector<int>> dataset0_lens         = {{2, 2}, {3, 3}};
    std::vector<std::vector<int>> dataset0_strides      = {{2, 2}, {1, 1}};
    std::vector<std::vector<int>> dataset0_pads         = {{0, 0}, {1, 1}};
    std::vector<miopenIndexType_t> dataset0_index_types = {
        miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    // Generate cartesian product for dataset 0
    // This matches the original ctest test_pooling2d behavior (default dataset, dataset_id=0)
    // Filter invalid combinations at generation time instead of skipping at runtime
    for(const auto& input_dims : dataset0_inputs)
    {
        AddTestCasesForInput(input_dims,
                             dataset0_lens,
                             dataset0_strides,
                             dataset0_pads,
                             dataset0_index_types,
                             modes,
                             wsidx_values,
                             test_cases,
                             false,  // skip_wide_check=false for Dataset 0
                             true);  // apply_index_type_limits=true for Dataset 0
    }

    // Note: Dataset 1 (asymmetric) and Dataset 2 (wide window) are tested separately
    // via pooling2d_asymmetric.cpp and pooling2d_wide.cpp to maintain the same
    // structure as the original ctest implementation.

    std::cerr << "\n=== Dataset 0 (Standard) Test Case Generation Summary ===\n";
    std::cerr << "Total test cases generated: " << test_cases.size() << "\n";
    std::cerr << "Input shapes: " << dataset0_inputs.size() << "\n";
    std::cerr << "===========================================================\n\n";

    // Log all test configurations to a file for comparison with ctest
    // Format: input_dims[4] lens[2] pads[2] strides[2] index_type mode wsidx
    std::ofstream log_file("pooling2d_gtest_configs.txt");
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
    cached            = true;

    return test_cases;
}

} // anonymous namespace

// Derived classes for Dataset 0 (standard pooling)
class GPU_Pooling2d_FP32 : public pooling2d_gtest::Pooling2dBatchCommon<float>
{
};

class GPU_Pooling2d_FP16 : public pooling2d_gtest::Pooling2dBatchCommon<half_float::half>
{
};

TEST_P(GPU_Pooling2d_FP32, FloatTest_pooling2d) { RunBatch(); }

TEST_P(GPU_Pooling2d_FP16, HalfTest_pooling2d) { RunBatch(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Pooling2d_FP32,
                         testing::ValuesIn(pooling2d_gtest::BatchTestCases(GetPooling2dTestCases(), 50)));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Pooling2d_FP16,
                         testing::ValuesIn(pooling2d_gtest::BatchTestCases(GetPooling2dTestCases(), 50)));
// -----------------------------------------------------------------------------
// Driver-based Full tests (kept from conversion commit).
// These are namespaced and renamed to avoid symbol collisions with the Smoke tests.
// -----------------------------------------------------------------------------

#include <miopen/env.hpp>
#include "get_handle.hpp"
#include "../pooling2d.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLAGS_ARGS)

namespace pooling2d_driver_ns {

namespace env = miopen::env;

class GPU_Pooling2dDriver_FP32 : public testing::TestWithParam<std::string>
{
    MIOPEN_DECLARE_GTEST_USES_TEST_DRIVE();
};

class GPU_Pooling2dDriver_FP16 : public testing::TestWithParam<std::string>
{
    MIOPEN_DECLARE_GTEST_USES_TEST_DRIVE();
};

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

void Run2dDriver(miopenDataType_t prec)
{
    std::string param;
    switch(prec)
    {
    case miopenFloat: param = GPU_Pooling2dDriver_FP32::GetParam(); break;
    case miopenHalf: param = GPU_Pooling2dDriver_FP16::GetParam(); break;
    default: param = GPU_Pooling2dDriver_FP32::GetParam();
    }

    std::vector<std::string> tokens;
    GetArgs(param, tokens);
    std::vector<const char*> ptrs;

    std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
        return str.data();
    });

    testing::internal::CaptureStderr();
    test_drive<pooling2d_driver<float>>(ptrs.size(), ptrs.data());
    auto capture = testing::internal::GetCapturedStderr();
    std::cout << capture;
}

bool IsTestSupportedForDevice(const miopen::Handle& handle) { return true; }

std::vector<std::string> GetTestCases(const std::string& precision)
{
    const auto& flag_arg = env::value(MIOPEN_TEST_FLAGS_ARGS);

    const std::vector<std::string> test_cases = {
        // clang-format off
        {"test_pooling2d " + precision + " --all --limit 0 " + flag_arg}
        // clang-format on
    };

    return test_cases;
}

} // namespace pooling2d_driver_ns

using namespace pooling2d_driver_ns;

TEST_P(GPU_Pooling2dDriver_FP32, FloatTest_pooling2d)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle))
    {
        Run2dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
}

TEST_P(GPU_Pooling2dDriver_FP16, HalfTest_pooling2d)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle))
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
}

INSTANTIATE_TEST_SUITE_P(Full, GPU_Pooling2dDriver_FP32, testing::ValuesIn(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(Full, GPU_Pooling2dDriver_FP16, testing::ValuesIn(GetTestCases("--half")));
