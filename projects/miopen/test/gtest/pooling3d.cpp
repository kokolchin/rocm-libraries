// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <half/half.hpp>

#include <miopen/logger.hpp>

#include "pooling_common.hpp"
#include "pooling2d_common.hpp"
#include "get_handle.hpp"
#include "gtest_common.hpp"

namespace {

using PoolingTestCase = pooling2d_gtest::PoolingTestCase;

std::vector<PoolingTestCase> GetPooling3dTestCases()
{
    static std::vector<PoolingTestCase> cached_test_cases;
    static bool cached = false;

    if(cached)
    {
        return cached_test_cases;
    }

    std::vector<PoolingTestCase> test_cases;

    // Dataset 0: Default dataset (various tensor sizes)
    // Match ctest generate_data_limited(..., 4)
    std::vector<std::vector<int>> dataset0_inputs = {
        {16, 64, 3, 4, 4}, {16, 32, 4, 9, 9}, {8, 512, 3, 14, 14}, {8, 512, 4, 28, 28}};

    std::vector<std::vector<int>> dataset0_lens    = {{2, 2, 2}, {3, 3, 3}};
    std::vector<std::vector<int>> dataset0_strides = {{2, 2, 2}, {1, 1, 1}};
    std::vector<std::vector<int>> dataset0_pads    = {{0, 0, 0}, {1, 1, 1}};
    std::vector<miopenIndexType_t> dataset0_index_types = {
        miopenIndexUint8, miopenIndexUint16, miopenIndexUint32, miopenIndexUint64};
    std::vector<miopenPoolingMode_t> modes = {
        miopenPoolingMax, miopenPoolingAverage, miopenPoolingAverageInclusive};
    std::vector<int> wsidx_values = {0, 1};

    // Generate cartesian product for dataset 0
    int num_uint16_case        = 0;
    int num_uint32_case        = 0;
    int num_uint32_case_imgidx = 0;
    int num_uint64_case        = 0;
    int num_uint64_case_imgidx = 0;

    for(const auto& input_dims : dataset0_inputs)
    {
        pooling2d_gtest::AddTestCasesForInput(input_dims,
                                              dataset0_lens,
                                              dataset0_strides,
                                              dataset0_pads,
                                              dataset0_index_types,
                                              modes,
                                              wsidx_values,
                                              test_cases,
                                              num_uint16_case,
                                              num_uint32_case,
                                              num_uint32_case_imgidx,
                                              num_uint64_case,
                                              num_uint64_case_imgidx,
                                              false,
                                              true);
    }

    // Dataset 1: Minimal dataset (asymmetric configs, small tensors)
    std::vector<std::vector<int>> dataset1_inputs  = {{1, 4, 4, 4, 4}};
    std::vector<std::vector<int>> dataset1_lens    = {{2, 2, 2}, {1, 2, 2}, {2, 1, 2}, {2, 2, 1}};
    std::vector<std::vector<int>> dataset1_strides = {
        {1, 1, 1}, {2, 1, 1}, {1, 2, 1}, {1, 1, 2}, {2, 2, 2}};
    std::vector<std::vector<int>> dataset1_pads         = {{0, 0, 0}};
    std::vector<miopenIndexType_t> dataset1_index_types = {miopenIndexUint8, miopenIndexUint32};

    // Generate cartesian product for dataset 1
    for(const auto& input_dims : dataset1_inputs)
    {
        pooling2d_gtest::AddTestCasesForInput(input_dims,
                                              dataset1_lens,
                                              dataset1_strides,
                                              dataset1_pads,
                                              dataset1_index_types,
                                              modes,
                                              wsidx_values,
                                              test_cases,
                                              num_uint16_case,
                                              num_uint32_case,
                                              num_uint32_case_imgidx,
                                              num_uint64_case,
                                              num_uint64_case_imgidx,
                                              true,
                                              false);
    }

    // Cache the results
    cached_test_cases = test_cases;
    cached            = true;

    return test_cases;
}

template <typename T, typename Index>
void RunPooling3dTestWithIndexType(const PoolingTestCase& test_case)
{
    // Create input tensor
    tensor<T> input{test_case.input_dims};
    input.generate(tensor_elem_gen_integer{miopen_type<T>{} == miopenHalf ? 5 : 17});

    // Setup pooling descriptor
    miopen::PoolingDescriptor filter{
        test_case.mode, miopenPaddingDefault, test_case.lens, test_case.strides, test_case.pads};
    filter.SetIndexType(test_case.index_type);
    filter.SetWorkspaceIndexMode(miopenPoolingWorkspaceIndexMode_t(test_case.wsidx));

    // Additional check: Skip if index_max is insufficient for output spatial dimensions (wsidx ==
    // 1) This check requires creating the filter and calculating output tensor, so it's done here
    if(test_case.mode == miopenPoolingMax && test_case.wsidx == 1)
    {
        // Calculate index_max based on index type from test_case
        size_t index_max = 0;
        switch(test_case.index_type)
        {
        case miopenIndexUint8: index_max = std::numeric_limits<uint8_t>::max(); break;
        case miopenIndexUint16: index_max = std::numeric_limits<uint16_t>::max(); break;
        case miopenIndexUint32: index_max = std::numeric_limits<uint32_t>::max(); break;
        case miopenIndexUint64: index_max = std::numeric_limits<uint64_t>::max(); break;
        default:
            index_max = SIZE_MAX; // Unknown type, assume it's large enough
            break;
        }

        // Check if index_max is sufficient for output spatial dimensions
        auto output_tensor            = get_output_tensor(filter, input);
        size_t output_spatial_product = 1;
        for(size_t i = 2; i < output_tensor.desc.GetLengths().size(); i++)
        {
            output_spatial_product *= static_cast<size_t>(output_tensor.desc.GetLengths()[i]);
        }
        if(index_max <= output_spatial_product)
        {
            int index_bits = 0;
            switch(test_case.index_type)
            {
            case miopenIndexUint8: index_bits = 8; break;
            case miopenIndexUint16: index_bits = 16; break;
            case miopenIndexUint32: index_bits = 32; break;
            case miopenIndexUint64: index_bits = 64; break;
            default: index_bits = 0; break;
            }
            GTEST_SKIP() << "Index range not enough: uint" << index_bits << " index_max ("
                         << index_max << ") <= output spatial product (" << output_spatial_product
                         << ") for max pooling backward with workspace index image mode";
        }
    }

    // Run forward pooling
    std::vector<Index> indices;
    verify_forward_pooling<3> forward_verifier;
    auto forward_result     = forward_verifier.cpu(input, filter, indices);
    auto forward_gpu_result = forward_verifier.gpu(input, filter, indices);

    // Compare forward results
    EXPECT_EQ(miopen::range_distance(forward_result), miopen::range_distance(forward_gpu_result));

    using value_type               = T;
    const double tolerance         = 80.0;
    const double threshold         = std::numeric_limits<value_type>::epsilon() * tolerance;
    const double forward_rms_error = miopen::rms_range(forward_result, forward_gpu_result);

    EXPECT_LE(forward_rms_error, threshold)
        << "Forward RMS error: " << forward_rms_error << " exceeds threshold: " << threshold;

    // Run backward pooling
    auto dout = forward_result;
    dout.generate(tensor_elem_gen_integer{2503});

    // Validate indices are populated (required for max pooling backward)
    if(test_case.mode == miopenPoolingMax && indices.empty())
    {
        GTEST_SKIP() << "Indices not populated for max pooling backward";
    }

    verify_backward_pooling<3> backward_verifier;
    auto backward_result = backward_verifier.cpu(
        input, dout, forward_result, filter, indices, test_case.wsidx != 0, true);
    auto backward_gpu_result = backward_verifier.gpu(
        input, dout, forward_result, filter, indices, test_case.wsidx != 0, true);

    // Compare backward results
    EXPECT_EQ(miopen::range_distance(backward_result), miopen::range_distance(backward_gpu_result));

    const double backward_rms_error = miopen::rms_range(backward_result, backward_gpu_result);

    EXPECT_LE(backward_rms_error, threshold)
        << "Backward RMS error: " << backward_rms_error << " exceeds threshold: " << threshold;
}

template <typename T>
void RunPooling3dTest(const PoolingTestCase& test_case)
{
    try
    {
        // Dispatch to the appropriate index type template
        switch(test_case.index_type)
        {
        case miopenIndexUint8: {
            RunPooling3dTestWithIndexType<T, uint8_t>(test_case);
            break;
        }
        case miopenIndexUint16: {
            RunPooling3dTestWithIndexType<T, uint16_t>(test_case);
            break;
        }
        case miopenIndexUint32: {
            RunPooling3dTestWithIndexType<T, uint32_t>(test_case);
            break;
        }
        case miopenIndexUint64: {
            RunPooling3dTestWithIndexType<T, uint64_t>(test_case);
            break;
        }
        default: {
            GTEST_FAIL() << "Unsupported index type: " << test_case.index_type;
            break;
        }
        }
    }
    catch(const std::exception& e)
    {
        std::string error_msg = e.what();
        // Skip test if no solver is found (unsupported configuration)
        if(error_msg.find("No solver found") != std::string::npos)
        {
            GTEST_SKIP() << "No solver found for test case: " << test_case;
        }
        GTEST_FAIL() << "Exception thrown with test case: " << test_case << "\n"
                     << "Exception: " << error_msg;
    }
    catch(...)
    {
        GTEST_FAIL() << "Unknown exception thrown with test case: " << test_case;
    }
}

} // namespace

// Helper function to estimate memory requirements for a test case
// Returns estimated memory in bytes needed for input, output, and workspace
size_t EstimateMemoryRequirements(const PoolingTestCase& test_case, size_t element_size)
{
    // Calculate input tensor size
    size_t input_size = element_size;
    for(int dim : test_case.input_dims)
    {
        input_size *= static_cast<size_t>(dim);
    }

    // Estimate output tensor size based on pooling parameters
    // Output spatial dimensions: floor((input + 2*pad - lens) / stride) + 1
    size_t output_spatial = 1;
    for(int i = 0; i < 3; i++)
    {
        int out_dim = (test_case.input_dims[i + 2] + 2 * test_case.pads[i] - test_case.lens[i]) /
                          test_case.strides[i] +
                      1;
        output_spatial *= static_cast<size_t>(out_dim);
    }
    size_t output_size = element_size * static_cast<size_t>(test_case.input_dims[0]) *
                         static_cast<size_t>(test_case.input_dims[1]) * output_spatial;

    // For max pooling, add workspace for indices
    size_t workspace_size = 0;
    if(test_case.mode == miopenPoolingMax)
    {
        // Index workspace: depends on wsidx mode
        if(test_case.wsidx == 0)
        {
            // Workspace index mask: size of pooling window per output element
            size_t window_size = 1;
            for(int len : test_case.lens)
            {
                window_size *= static_cast<size_t>(len);
            }
            // Index type size
            size_t index_size = 1;
            switch(test_case.index_type)
            {
            case miopenIndexUint8: index_size = 1; break;
            case miopenIndexUint16: index_size = 2; break;
            case miopenIndexUint32: index_size = 4; break;
            case miopenIndexUint64: index_size = 8; break;
            default: index_size = 4; break;
            }
            workspace_size = output_size * window_size * index_size / element_size;
        }
        else
        {
            // Workspace index image: one index per output element
            size_t index_size = 1;
            switch(test_case.index_type)
            {
            case miopenIndexUint8: index_size = 1; break;
            case miopenIndexUint16: index_size = 2; break;
            case miopenIndexUint32: index_size = 4; break;
            case miopenIndexUint64: index_size = 8; break;
            default: index_size = 4; break;
            }
            workspace_size = output_size * index_size / element_size;
        }
    }

    // Total estimate: input + output + workspace + overhead (1.5x for safety)
    return static_cast<size_t>((input_size + output_size + workspace_size) * 1.5);
}

// Helper function to perform early skip checks that don't require tensor creation
void CheckPooling3dTestCase(const PoolingTestCase& test_case)
{
    // Validate dimensions
    int spt_dim = static_cast<int>(test_case.input_dims.size()) - 2;

    if(spt_dim != 3)
    {
        GTEST_SKIP() << "Only 3D pooling is supported (spt_dim == 3)";
    }

    // Estimate memory requirements and skip if too large
    // Conservative threshold: 1.5 GB to prevent out-of-memory errors on GPUs with limited memory
    const size_t memory_threshold_bytes = 1500ULL * 1024 * 1024; // 1.5 GB
    size_t estimated_memory_fp32        = EstimateMemoryRequirements(test_case, sizeof(float));
    size_t estimated_memory_fp16 = EstimateMemoryRequirements(test_case, 2); // half is 2 bytes

    if(estimated_memory_fp32 > memory_threshold_bytes ||
       estimated_memory_fp16 > memory_threshold_bytes)
    {
        GTEST_SKIP() << "Test case requires too much memory (estimated: "
                     << (estimated_memory_fp32 / (1024 * 1024)) << " MB for FP32, "
                     << (estimated_memory_fp16 / (1024 * 1024))
                     << " MB for FP16). Skipping to avoid OOM.";
    }

    // Check kernel size vs input dimensions
    for(int i = 0; i < spt_dim; i++)
    {
        if(test_case.lens[i] > (static_cast<uint64_t>(test_case.input_dims[i + 2]) +
                                static_cast<uint64_t>(2) * test_case.pads[i]))
        {
            GTEST_SKIP() << "Invalid config: lens[" << i << "] > (input_dims[" << i + 2
                         << "] + 2 * pads[" << i << "])";
        }
    }

    // Skip configurations that would cause "Index range not enough" exception
    // The original ctest skips ALL uint8/uint16 max pooling in 3D
    // (matching the original ctest behavior: spt_dim == 3 && mode == Max)
    if(test_case.mode == miopenPoolingMax &&
       (test_case.index_type == miopenIndexUint8 || test_case.index_type == miopenIndexUint16))
    {
        GTEST_SKIP() << "Config skipped: uint"
                     << (test_case.index_type == miopenIndexUint8 ? 8 : 16)
                     << " index is too small (spt_dim == 3 && mode == Max)";
    }

    // Check if index_max is insufficient for the pooling window (for wsidx == 0)
    if(test_case.mode == miopenPoolingMax && test_case.wsidx == 0)
    {
        size_t index_max = 0;
        switch(test_case.index_type)
        {
        case miopenIndexUint8: index_max = std::numeric_limits<uint8_t>::max(); break;
        case miopenIndexUint16: index_max = std::numeric_limits<uint16_t>::max(); break;
        case miopenIndexUint32: index_max = std::numeric_limits<uint32_t>::max(); break;
        case miopenIndexUint64: index_max = std::numeric_limits<uint64_t>::max(); break;
        default: break;
        }

        size_t lens_product = 1;
        for(int len : test_case.lens)
        {
            lens_product *= static_cast<size_t>(len);
        }
        if(index_max > 0 && index_max <= lens_product)
        {
            int index_bits = 0;
            switch(test_case.index_type)
            {
            case miopenIndexUint8: index_bits = 8; break;
            case miopenIndexUint16: index_bits = 16; break;
            case miopenIndexUint32: index_bits = 32; break;
            case miopenIndexUint64: index_bits = 64; break;
            default: index_bits = 0; break;
            }
            GTEST_SKIP() << "Index range not enough: uint" << index_bits << " index_max ("
                         << index_max << ") <= lens product (" << lens_product
                         << ") for max pooling backward with workspace index mask mode";
        }
    }
}

class GPU_Pooling3d_FP32 : public testing::TestWithParam<pooling2d_gtest::PoolingBatch>
{
    void SetUp() override
    {
        prng::reset_seed();
    }
};

class GPU_Pooling3d_FP16 : public testing::TestWithParam<pooling2d_gtest::PoolingBatch>
{
    void SetUp() override
    {
        prng::reset_seed();
    }
};

TEST_P(GPU_Pooling3d_FP32, Test) 
{ 
    for(const auto& tc : GetParam().test_cases)
    {
        CheckPooling3dTestCase(tc);
        RunPooling3dTest<float>(tc); 
    }
}

TEST_P(GPU_Pooling3d_FP16, Test) 
{ 
    for(const auto& tc : GetParam().test_cases)
    {
        CheckPooling3dTestCase(tc);
        RunPooling3dTest<half_float::half>(tc); 
    }
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Pooling3d_FP32,
                         testing::ValuesIn(pooling2d_gtest::BatchTestCases(GetPooling3dTestCases(), 50)));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Pooling3d_FP16,
                         testing::ValuesIn(pooling2d_gtest::BatchTestCases(GetPooling3dTestCases(), 50)));
