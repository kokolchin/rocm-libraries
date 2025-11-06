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

#include <miopen/batch_norm.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "get_handle.hpp"
#include "tensor_holder.hpp"

namespace {

#define MIO_BN_TEST_EPSILON 1e-5
#define MIO_BN_TEST_EXPAVGFACTOR 0.1

// Simplified 3D BN PerActivation test fixture
class BN3DPerActTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Simple 3D tensor: batch=4, channels=2, depth=3, height=8, width=8
        n = 4;
        c = 2;
        d = 3;
        h = 8;
        w = 8;

        auto&& h = get_handle();

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
        in_dev      = h.Write(input.data);
        scale_dev   = h.Write(scale.data);
        shift_dev   = h.Write(shift.data);
        runMean_dev = h.Write(runMean.data);
        runVar_dev  = h.Write(runVar.data);
        out_dev     = h.Write(output.data);
    }

    auto get_handle_ref() -> miopen::Handle& { return get_handle(); }
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

// Test forward training
TEST_F(BN3DPerActTest, ForwardTraining)
{
    // Create saved mean and variance for training
    tensor<float> savedMean{1, c, d, h, w};
    tensor<float> savedInvVar{1, c, d, h, w};
    auto&& handle        = get_handle_ref();
    auto savedMean_dev   = handle.Write(savedMean.data);
    auto savedInvVar_dev = handle.Write(savedInvVar.data);

    miopenStatus_t status = miopenBatchNormForwardTraining(&handle,
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
}

// Test forward inference (recalc)
TEST_F(BN3DPerActTest, ForwardInferenceRecalc)
{
    auto&& handle         = get_handle_ref();
    miopenStatus_t status = miopenBatchNormForwardInference(&handle,
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
}

// Test forward inference (use estimated)
TEST_F(BN3DPerActTest, ForwardInferenceUseEstimated)
{
    auto&& handle = get_handle_ref();
    // Use the running mean and variance as estimated values
    miopenStatus_t status = miopenBatchNormForwardInference(&handle,
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
}

// Test backward (recalc)
TEST_F(BN3DPerActTest, BackwardRecalc)
{
    // Create dy input (gradient from next layer)
    tensor<float> dy_input{n, c, d, h, w};
    dy_input.generate([](int n, int c, int d, int h, int w) {
        return static_cast<float>((n * 50 + c * 5 + d + h + w) % 13) * 0.01f;
    });
    auto&& handle = get_handle_ref();
    auto dy_dev   = handle.Write(dy_input.data);

    // Outputs for backward
    tensor<float> dx_output{n, c, d, h, w};
    tensor<float> dscale{1, c, d, h, w};
    tensor<float> dshift{1, c, d, h, w};
    auto dx_dev     = handle.Write(dx_output.data);
    auto dscale_dev = handle.Write(dscale.data);
    auto dshift_dev = handle.Write(dshift.data);

    miopenStatus_t status = miopenBatchNormBackward(&handle,
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
}

// Test backward (use saved)
TEST_F(BN3DPerActTest, BackwardUseSaved)
{
    // First do forward training to get saved values
    tensor<float> savedMean{1, c, d, h, w};
    tensor<float> savedInvVar{1, c, d, h, w};
    auto&& handle        = get_handle_ref();
    auto savedMean_dev   = handle.Write(savedMean.data);
    auto savedInvVar_dev = handle.Write(savedInvVar.data);

    miopenStatus_t status = miopenBatchNormForwardTraining(&handle,
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
    auto&& handle = get_handle_ref();
    auto dy_dev   = handle.Write(dy_input.data);

    tensor<float> dx_output{n, c, d, h, w};
    tensor<float> dscale{1, c, d, h, w};
    tensor<float> dshift{1, c, d, h, w};
    auto dx_dev     = handle.Write(dx_output.data);
    auto dscale_dev = handle.Write(dscale.data);
    auto dshift_dev = handle.Write(dshift.data);

    status = miopenBatchNormBackward(&handle,
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
}

} // namespace
