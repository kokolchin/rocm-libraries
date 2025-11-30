<<<<<<< HEAD
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <vector>
#include <gtest/gtest.h>
#include <half/half.hpp>
// network_data.hpp provides get_inputs() function used when TEST_GET_INPUT_TENSOR = 1
// (currently TEST_GET_INPUT_TENSOR = 0, but include is needed to support both cases)
#include "../network_data.hpp"
#include "pooling2d_common.hpp"

using PoolingTestCase = pooling2d_gtest::PoolingTestCase;

// TEST_GET_INPUT_TENSOR is defined in pooling2d_common.hpp
// When 0: uses all 18 predefined input shapes (matching ctest with --all)
// When 1: uses get_inputs() function to generate input shapes from network_data

namespace {

using namespace pooling2d_gtest;

std::vector<PoolingTestCase> GetPooling2dTestCases()
{
    // Cache results to avoid duplicate generation when called multiple times
    // (e.g., for both FP32 and FP16 test instantiations)
    static std::vector<PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<PoolingTestCase> test_cases;

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
    std::vector<std::vector<int>> dataset0_lens = {{2, 2}, {3, 3}};
    // Note: Order matters for index type limits! CTest processes stride (2,2) before (1,1)
    std::vector<std::vector<int>> dataset0_strides      = {{2, 2}, {1, 1}};
    std::vector<std::vector<int>> dataset0_pads         = {{0, 0}, {1, 1}};
    std::vector<miopenIndexType_t> dataset0_index_types = {
        miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    // Generate cartesian product for dataset 0
    // This matches the original ctest test_pooling2d behavior (default dataset, dataset_id=0)
    // IMPORTANT: Order must match ctest exactly: index_type -> mode -> input_shape -> lens ->
    // strides -> pads -> wsidx This is the order in which test_driver processes test cases (based
    // on add() call order)
    for(const auto& index_type : dataset0_index_types)
    {
        for(const auto& mode : modes)
        {
            for(const auto& input_dims : dataset0_inputs)
            {
                AddTestCasesForInput(input_dims,
                                     dataset0_lens,
                                     dataset0_strides,
                                     dataset0_pads,
                                     {index_type}, // Single index_type for this iteration
                                     {mode},       // Single mode for this iteration
                                     wsidx_values,
                                     test_cases,
                                     false,  // skip_wide_check=false for Dataset 0
                                     true,   // apply_index_type_limits=true for Dataset 0
                                     false); // is_wide_dataset=false for Dataset 0
            }
        }
    }

    // Note: Dataset 1 (asymmetric) and Dataset 2 (wide window) are tested separately
    // via pooling2d_asymmetric.cpp and pooling2d_wide.cpp to maintain the same
    // structure as the original ctest implementation.

    // Cache the results
    cached_test_cases = test_cases;
    cached            = true;
=======
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "pooling2d.hpp"

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLAGS_ARGS)

namespace env = miopen::env;

namespace pooling2d {

class GPU_Pooling2d_FP32 : public testing::TestWithParam<std::string>
{
    MIOPEN_DECLARE_GTEST_USES_TEST_DRIVE();
};

class GPU_Pooling2d_FP16 : public testing::TestWithParam<std::string>
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
    case miopenFloat: param = GPU_Pooling2d_FP32::GetParam(); break;
    case miopenHalf: param = GPU_Pooling2d_FP16::GetParam(); break;
    case miopenBFloat16:
    case miopenInt8:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
        FAIL() << "miopenBFloat16, miopenInt8, miopenInt32, miopenDouble, miopenFloat8_fnuz, "
                  "miopenBFloat8_fnuz "
                  "data type not supported by "
                  "pooling2d test";

    default: param = GPU_Pooling2d_FP32::GetParam();
    }

    std::vector<std::string> tokens;
    GetArgs(param, tokens);
    std::vector<const char*> ptrs;

    std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
        return str.data();
    });

    testing::internal::CaptureStderr();
    test_drive<pooling2d_driver>(ptrs.size(), ptrs.data());
    auto capture = testing::internal::GetCapturedStderr();
    std::cout << capture;
};

bool IsTestSupportedForDevice(const miopen::Handle& handle) { return true; }

std::vector<std::string> GetTestCases(const std::string& precision)
{
    const auto& flag_arg = env::value(MIOPEN_TEST_FLAGS_ARGS);

    const std::vector<std::string> test_cases = {
        // clang-format off
    {"test_pooling2d " + precision + " --all --limit 0 " + flag_arg}
        // clang-format on
    };
>>>>>>> f992f34d7e (Convert pooling2d.cpp from ctest to Google Test)

    return test_cases;
}

<<<<<<< HEAD
} // namespace

// Derived classes for Dataset 0 (standard pooling)
using GPU_Pooling2d_FP32 = Pooling2dCommon<float>;
using GPU_Pooling2d_FP16 = Pooling2dCommon<half_float::half>;

TEST_P(GPU_Pooling2d_FP32, FloatTest_pooling2d) { RunTest(); }

TEST_P(GPU_Pooling2d_FP16, HalfTest_pooling2d) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Pooling2d_FP32,
                         testing::ValuesIn(GetPooling2dTestCases()),
                         GetPoolingTestCaseName);

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Pooling2d_FP16,
                         testing::ValuesIn(GetPooling2dTestCases()),
                         GetPoolingTestCaseName);
=======
} // namespace pooling2d
using namespace pooling2d;

TEST_P(GPU_Pooling2d_FP32, FloatTest_pooling2d)
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
};

TEST_P(GPU_Pooling2d_FP16, HalfTest_pooling2d)
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
};

INSTANTIATE_TEST_SUITE_P(Full, GPU_Pooling2d_FP32, testing::ValuesIn(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(Full, GPU_Pooling2d_FP16, testing::ValuesIn(GetTestCases("--half")));

>>>>>>> f992f34d7e (Convert pooling2d.cpp from ctest to Google Test)
