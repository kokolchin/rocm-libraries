// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/batch_norm.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "get_handle.hpp"
#include "tensor_holder.hpp"

namespace {
constexpr double MIO_BN_TEST_EPSILON      = 1e-5;
constexpr double MIO_BN_TEST_EXPAVGFACTOR = 0.1;

enum class BN3DPerActTestType
{
    ForwardTraining,
    ForwardInferenceRecalc,
    ForwardInferenceUseEstimated,
    BackwardRecalc,
    BackwardUseSaved
};

struct BN3DPerActTestCase
{
    BN3DPerActTestType test_type;
};

std::vector<BN3DPerActTestCase> GetBN3DPerActTestCases()
{
    return {{BN3DPerActTestType::ForwardTraining},
            {BN3DPerActTestType::ForwardInferenceRecalc},
            {BN3DPerActTestType::ForwardInferenceUseEstimated},
            {BN3DPerActTestType::BackwardRecalc},
            {BN3DPerActTestType::BackwardUseSaved}};
}
} // namespace

struct GPU_Bn3dPerAct_FP32 : public ::testing::TestWithParam<BN3DPerActTestCase>
{
    void SetUp() override
    {
        // Reset internal environment values to ensure tests are order-agnostic
        // See: https://github.com/ROCm/MIOpen/wiki/GTest-development
        // Note: This test does not use PRNG or other internal environment values that need resetting

        // Simple 3D tensor: batch=4, channels=2, depth=3, height=8, width=8
        n = 4;
        c = 2;
        d = 3;
        h = 8;
        w = 8;

        auto&& handle = get_handle();

        // Create input tensor
        input = tensor<float>{n, c, d, h, w};
        input.generate([](int n, int c, int d, int h, int w) {
            return static_cast<float>((n * 100 + c * 10 + d + h + w) % 17);
        });

        // Derive BN descriptor for PerActivation mode
        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNPerActivation);

        std::size_t ssn, ssc, ssd, ssh, ssw;
        std::tie(ssn, ssc, ssd, ssh, ssw) = miopen::tien<5>(derivedBnDesc.GetLengths());

        // Create scale and shift tensors
        scale = tensor<float>{ssn, ssc, ssd, ssh, ssw};
        shift = tensor<float>{ssn, ssc, ssd, ssh, ssw};

        for(std::size_t i = 0; i < scale.desc.GetElementSize(); i++)
        {
            scale[i] = 1.0f;
            shift[i] = 0.0f;
        }

        // Initialize running mean and variance
        runMean = tensor<float>{ssn, ssc, ssd, ssh, ssw};
        runVar  = tensor<float>{ssn, ssc, ssd, ssh, ssw};

        for(std::size_t i = 0; i < runMean.desc.GetElementSize(); i++)
        {
            runMean[i] = 0.0f;
            runVar[i]  = 1.0f;
        }

        // Create output tensor
        output = tensor<float>{n, c, d, h, w};

        // Allocate GPU memory
        in_dev      = handle.Write(input.data);
        scale_dev   = handle.Write(scale.data);
        shift_dev   = handle.Write(shift.data);
        runMean_dev = handle.Write(runMean.data);
        runVar_dev  = handle.Write(runVar.data);
        out_dev     = handle.Write(output.data);
    }

    std::size_t n, c, d, h, w;
    tensor<float> input;
    tensor<float> output;
    tensor<float> scale;
    tensor<float> shift;
    tensor<float> runMean;
    tensor<float> runVar;
    miopen::TensorDescriptor derivedBnDesc;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr scale_dev;
    miopen::Allocator::ManageDataPtr shift_dev;
    miopen::Allocator::ManageDataPtr runMean_dev;
    miopen::Allocator::ManageDataPtr runVar_dev;
    miopen::Allocator::ManageDataPtr out_dev;

    float alpha         = 1.0f;
    float beta          = 0.0f;
    double epsilon      = MIO_BN_TEST_EPSILON;
    double expAvgFactor = MIO_BN_TEST_EXPAVGFACTOR;
};

TEST_P(GPU_Bn3dPerAct_FP32, Test)
{
    const auto& test_case = this->GetParam();
    auto&& handle         = get_handle();

    switch(test_case.test_type)
    {
    case BN3DPerActTestType::ForwardTraining: {
        // Create saved mean and variance for training
        tensor<float> savedMean{1, c, d, h, w};
        tensor<float> savedInvVar{1, c, d, h, w};
        auto savedMean_dev   = handle.Write(savedMean.data);
        auto savedInvVar_dev = handle.Write(savedInvVar.data);

        miopenStatus_t status = miopenBatchNormalizationForwardTraining(&handle,
                                                                        miopenBNPerActivation,
                                                                        &alpha,
                                                                        &beta,
                                                                        &input.desc,
                                                                        in_dev.get(),
                                                                        &output.desc,
                                                                        out_dev.get(),
                                                                        &derivedBnDesc,
                                                                        scale_dev.get(),
                                                                        shift_dev.get(),
                                                                        expAvgFactor,
                                                                        runMean_dev.get(),
                                                                        runVar_dev.get(),
                                                                        epsilon,
                                                                        savedMean_dev.get(),
                                                                        savedInvVar_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);
        break;
    }
    case BN3DPerActTestType::ForwardInferenceRecalc:
    case BN3DPerActTestType::ForwardInferenceUseEstimated: {
        miopenStatus_t status = miopenBatchNormalizationForwardInference(&handle,
                                                                         miopenBNPerActivation,
                                                                         &alpha,
                                                                         &beta,
                                                                         &input.desc,
                                                                         in_dev.get(),
                                                                         &output.desc,
                                                                         out_dev.get(),
                                                                         &derivedBnDesc,
                                                                         scale_dev.get(),
                                                                         shift_dev.get(),
                                                                         runMean_dev.get(),
                                                                         runVar_dev.get(),
                                                                         epsilon);

        EXPECT_EQ(status, miopenStatusSuccess);
        break;
    }
    case BN3DPerActTestType::BackwardRecalc: {
        // Create dy input (gradient from next layer)
        tensor<float> dy_input{n, c, d, h, w};
        dy_input.generate([](int n, int c, int d, int h, int w) {
            return static_cast<float>((n * 50 + c * 5 + d + h + w) % 13) * 0.01f;
        });
        auto dy_dev = handle.Write(dy_input.data);

        // Outputs for backward
        tensor<float> dx_output{n, c, d, h, w};
        tensor<float> dscale{1, c, d, h, w};
        tensor<float> dshift{1, c, d, h, w};
        auto dx_dev     = handle.Write(dx_output.data);
        auto dscale_dev = handle.Write(dscale.data);
        auto dshift_dev = handle.Write(dshift.data);

        miopenStatus_t status =
            miopenBatchNormalizationBackward(&handle,
                                             miopenBNPerActivation,
                                             &alpha,
                                             &beta,
                                             &alpha,
                                             &beta,
                                             &input.desc,
                                             in_dev.get(),
                                             &dy_input.desc,
                                             dy_dev.get(),
                                             &dx_output.desc,
                                             dx_dev.get(),
                                             &derivedBnDesc,
                                             scale_dev.get(),
                                             dscale_dev.get(),
                                             dshift_dev.get(),
                                             epsilon,
                                             nullptr, // savedMean - nullptr means recalc
                                             nullptr  // savedInvVar - nullptr means recalc
            );

        EXPECT_EQ(status, miopenStatusSuccess);
        break;
    }
    case BN3DPerActTestType::BackwardUseSaved: {
        // First do forward training to get saved values
        tensor<float> savedMean{1, c, d, h, w};
        tensor<float> savedInvVar{1, c, d, h, w};
        auto savedMean_dev   = handle.Write(savedMean.data);
        auto savedInvVar_dev = handle.Write(savedInvVar.data);

        miopenStatus_t status = miopenBatchNormalizationForwardTraining(&handle,
                                                                        miopenBNPerActivation,
                                                                        &alpha,
                                                                        &beta,
                                                                        &input.desc,
                                                                        in_dev.get(),
                                                                        &output.desc,
                                                                        out_dev.get(),
                                                                        &derivedBnDesc,
                                                                        scale_dev.get(),
                                                                        shift_dev.get(),
                                                                        expAvgFactor,
                                                                        runMean_dev.get(),
                                                                        runVar_dev.get(),
                                                                        epsilon,
                                                                        savedMean_dev.get(),
                                                                        savedInvVar_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        // Now do backward using saved values
        tensor<float> dy_input{n, c, d, h, w};
        dy_input.generate([](int n, int c, int d, int h, int w) {
            return static_cast<float>((n * 50 + c * 5 + d + h + w) % 13) * 0.01f;
        });
        auto dy_dev = handle.Write(dy_input.data);

        tensor<float> dx_output{n, c, d, h, w};
        tensor<float> dscale{1, c, d, h, w};
        tensor<float> dshift{1, c, d, h, w};
        auto dx_dev     = handle.Write(dx_output.data);
        auto dscale_dev = handle.Write(dscale.data);
        auto dshift_dev = handle.Write(dshift.data);

        status = miopenBatchNormalizationBackward(&handle,
                                                  miopenBNPerActivation,
                                                  &alpha,
                                                  &beta,
                                                  &alpha,
                                                  &beta,
                                                  &input.desc,
                                                  in_dev.get(),
                                                  &dy_input.desc,
                                                  dy_dev.get(),
                                                  &dx_output.desc,
                                                  dx_dev.get(),
                                                  &derivedBnDesc,
                                                  scale_dev.get(),
                                                  dscale_dev.get(),
                                                  dshift_dev.get(),
                                                  epsilon,
                                                  savedMean_dev.get(),  // use saved mean
                                                  savedInvVar_dev.get() // use saved inv var
        );

        EXPECT_EQ(status, miopenStatusSuccess);
        break;
    }
    }
}

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Bn3dPerAct_FP32, testing::ValuesIn(GetBN3DPerActTestCases()));
