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
#include "../random.hpp"
#include "../verify.hpp"
#include "../fusionHost.hpp"
#include "test_operations.hpp"
#include "network_data.hpp"

namespace {
constexpr double MIO_BN_TEST_EPSILON      = 1e-5;
constexpr double MIO_BN_TEST_EXPAVGFACTOR = 0.1;
constexpr double MIO_BN_TEST_TOLERANCE    = 0.5;

// Helper struct to provide a view of a tensor with a local descriptor copy
// to avoid modifying class members during CPU verification reshaping.


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
    std::size_t n, c, d, h, w;
    BN3DPerActTestType test_type;

    friend std::ostream& operator<<(std::ostream& ss, const BN3DPerActTestCase& tc)
    {
        return ss << "(n: " << tc.n << " c: " << tc.c << " d: " << tc.d << " h: " << tc.h
                  << " w: " << tc.w << " type: " << static_cast<int>(tc.test_type) << ")";
    }
};

std::vector<BN3DPerActTestCase> GetBN3DPerActTestCases()
{
    std::vector<BN3DPerActTestCase> test_cases;
    const std::vector<BN3DPerActTestType> types = {BN3DPerActTestType::ForwardTraining,
                                                   BN3DPerActTestType::ForwardInferenceRecalc,
                                                   BN3DPerActTestType::ForwardInferenceUseEstimated,
                                                   BN3DPerActTestType::BackwardRecalc,
                                                   BN3DPerActTestType::BackwardUseSaved};

    // Use batch size factor 4 to match ctest behavior
    for(const auto& shape : get_3d_bn_peract_inputs(4))
    {
        const auto n = shape[0];
        if(n == 1)
            continue;

        for(const auto& type : types)
            test_cases.push_back({shape[0], shape[1], shape[2], shape[3], shape[4], type});
    }
    return test_cases;
}

template <typename T>
struct GPU_Bn3dPerAct : public ::testing::TestWithParam<BN3DPerActTestCase>
{
    using AccDataType = std::conditional_t<std::is_same_v<T, double>, double, float>;

    static tensor<T> pool_input;
    static tensor<AccDataType> pool_scale;
    static tensor<AccDataType> pool_shift;
    static tensor<AccDataType> pool_runMean;
    static tensor<AccDataType> pool_runVar;

    static void SetUpTestSuite()
    {
        auto test_cases   = GetBN3DPerActTestCases();
        std::size_t max_n = 0, max_c = 0, max_d = 0, max_h = 0, max_w = 0;
        for(const auto& tc : test_cases)
        {
            max_n = std::max(max_n, tc.n);
            max_c = std::max(max_c, tc.c);
            max_d = std::max(max_d, tc.d);
            max_h = std::max(max_h, tc.h);
            max_w = std::max(max_w, tc.w);
        }

        // Pre-allocate and generate data for the largest possible shape
        // For 3D BN Per-Activation, derived tensors are 1xCxDxHxW
        pool_input = tensor<T>{miopenTensorNCDHW,
                               std::vector<std::size_t>{max_n, max_c, max_d, max_h, max_w}};
        pool_input.generate(uniform_signed_initializer<T>(2e-3, 1000));

        std::vector<std::size_t> derived_lens = {1, max_c, max_d, max_h, max_w};
        pool_scale   = tensor<AccDataType>{miopenTensorNCDHW, derived_lens};
        pool_shift   = tensor<AccDataType>{miopenTensorNCDHW, derived_lens};
        pool_runMean = tensor<AccDataType>{miopenTensorNCDHW, derived_lens};
        pool_runVar  = tensor<AccDataType>{miopenTensorNCDHW, derived_lens};

        pool_scale.generate(uniform_signed_initializer<AccDataType>(2e-3, 1000));
        pool_shift.generate(uniform_signed_initializer<AccDataType>(2e-3, 1000));
        pool_runMean.generate(uniform_signed_initializer<AccDataType>(2e-3, 1000));
        pool_runVar.generate(uniform_unsigned_initializer<AccDataType>(2e-3, 1000));
    }

    void SetUp() override
    {
        prng::reset_seed();
        const auto& tc = this->GetParam();
        n              = tc.n;
        c              = tc.c;
        d              = tc.d;
        h              = tc.h;
        w              = tc.w;

        // Match ctest logic: skip ALL test cases when n == 1
        if(n == 1)
        {
            GTEST_SKIP() << "Invalid batch size for batch norm tests";
        }

        auto&& handle = get_handle();

        input  = tensor<T>{miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w}};
        output = tensor<T>{miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w}};

        // Get layout from input before creating out_ref to ensure consistency
        auto input_layout_opt = input.desc.GetLayoutEnum();
        bn_layout = (input_layout_opt && input_layout_opt.value() != 0) ? input_layout_opt.value()
                                                                        : miopenTensorNCDHW;
        if(bn_layout == 0)
            bn_layout = miopenTensorNCDHW;

        out_ref = tensor<AccDataType>{bn_layout, std::vector<std::size_t>{n, c, d, h, w}};

        std::copy(pool_input.data.begin(),
                  pool_input.data.begin() + input.data.size(),
                  input.data.begin());

        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNPerActivation);
        auto derived_num_dims = derivedBnDesc.GetLengths().size();
        derived_layout        = (derived_num_dims == 5) ? miopenTensorNCDHW : miopenTensorNCHW;

        bool need_fix           = false;
        auto derived_layout_opt = derivedBnDesc.GetLayoutEnum();
        if(!derived_layout_opt || derived_layout_opt.value() != derived_layout)
        {
            need_fix = true;
        }
        else if(derived_num_dims == 4)
        {
            auto layout_val = derived_layout_opt.value();
            if(layout_val != miopenTensorNCHW && layout_val != miopenTensorNHWC &&
               layout_val != miopenTensorCHWN && layout_val != miopenTensorNCHWc4 &&
               layout_val != miopenTensorNCHWc8 && layout_val != miopenTensorCHWNc4 &&
               layout_val != miopenTensorCHWNc8)
                need_fix = true;
        }
        else if(derived_num_dims == 5)
        {
            auto layout_val = derived_layout_opt.value();
            if(layout_val != miopenTensorNCDHW && layout_val != miopenTensorNDHWC)
                need_fix = true;
        }
        else
        {
            need_fix = true;
        }

        if(need_fix)
        {
            derivedBnDesc = miopen::TensorDescriptor(
                derivedBnDesc.GetType(), derived_layout, derivedBnDesc.GetLengths());
        }

        scale   = tensor<AccDataType>{derived_layout, derivedBnDesc.GetLengths()};
        shift   = tensor<AccDataType>{derived_layout, derivedBnDesc.GetLengths()};
        runMean = tensor<AccDataType>{derived_layout, derivedBnDesc.GetLengths()};
        runVar  = tensor<AccDataType>{derived_layout, derivedBnDesc.GetLengths()};

        std::copy(pool_scale.data.begin(),
                  pool_scale.data.begin() + scale.data.size(),
                  scale.data.begin());
        std::copy(pool_shift.data.begin(),
                  pool_shift.data.begin() + shift.data.size(),
                  shift.data.begin());
        std::copy(pool_runMean.data.begin(),
                  pool_runMean.data.begin() + runMean.data.size(),
                  runMean.data.begin());
        std::copy(pool_runVar.data.begin(),
                  pool_runVar.data.begin() + runVar.data.size(),
                  runVar.data.begin());

        in_dev      = handle.Write(input.data);
        scale_dev   = handle.Write(scale.data);
        shift_dev   = handle.Write(shift.data);
        runMean_dev = handle.Write(runMean.data);
        runVar_dev  = handle.Write(runVar.data);
        out_dev     = handle.Write(output.data);

        tolerance = MIO_BN_TEST_TOLERANCE;
    }

    template <typename TensorType>
    static void EnsureValidLayout(TensorType& t, miopenTensorLayout_t default_layout)
    {
        auto layout_t = t.desc.GetLayout_t();
        auto num_dims = t.desc.GetLengths().size();

        miopenTensorLayout_t valid_layout;
        if(num_dims == 5)
            valid_layout = miopenTensorNCDHW;
        else if(num_dims == 4)
            valid_layout = miopenTensorNCHW;
        else
            valid_layout = default_layout;

        if(layout_t == 0 ||
           (num_dims == 5 && layout_t != miopenTensorNCDHW && layout_t != miopenTensorNDHWC) ||
           (num_dims == 4 && layout_t != miopenTensorNCHW && layout_t != miopenTensorNHWC &&
            layout_t != miopenTensorCHWN && layout_t != miopenTensorNCHWc4 &&
            layout_t != miopenTensorNCHWc8 && layout_t != miopenTensorCHWNc4 &&
            layout_t != miopenTensorCHWNc8))
        {
            t.desc = miopen::TensorDescriptor(t.desc.GetType(), valid_layout, t.desc.GetLengths());
        }
    }

    std::size_t n, c, d, h, w;
    tensor<T> input;
    tensor<T> output;
    tensor<AccDataType> out_ref;
    tensor<AccDataType> scale;
    tensor<AccDataType> shift;
    tensor<AccDataType> runMean;
    tensor<AccDataType> runVar;
    miopen::TensorDescriptor derivedBnDesc;
    miopenTensorLayout_t bn_layout;
    miopenTensorLayout_t derived_layout;
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
    double tolerance    = MIO_BN_TEST_TOLERANCE;

    void RunTest()
    {
        const auto& test_case = this->GetParam();

        if(test_case.test_type == BN3DPerActTestType::ForwardInferenceUseEstimated &&
           (std::is_same_v<T, half_float::half> || std::is_same_v<T, bfloat16>))
        {
            GTEST_SKIP() << "ForwardInferenceUseEstimated not supported for half precision 3D BN";
        }

        auto&& handle = get_handle();

        switch(test_case.test_type)
        {
        case BN3DPerActTestType::ForwardTraining: {
            tensor<AccDataType> saveMean{derived_layout, derivedBnDesc.GetLengths()};
            tensor<AccDataType> saveInvVar{derived_layout, derivedBnDesc.GetLengths()};
            auto saveMean_dev   = handle.Write(saveMean.data);
            auto saveInvVar_dev = handle.Write(saveInvVar.data);

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
                                                                            saveMean_dev.get(),
                                                                            saveInvVar_dev.get());

            ASSERT_EQ(status, miopenStatusSuccess);

            output.data     = handle.Read<T>(out_dev, output.data.size());
            saveMean.data   = handle.Read<AccDataType>(saveMean_dev, saveMean.data.size());
            saveInvVar.data = handle.Read<AccDataType>(saveInvVar_dev, saveInvVar.data.size());
            runMean.data    = handle.Read<AccDataType>(runMean_dev, runMean.data.size());
            runVar.data     = handle.Read<AccDataType>(runVar_dev, runVar.data.size());

            // Create separate tensors for CPU reference results
            tensor<AccDataType> saveMean_ref{derived_layout, derivedBnDesc.GetLengths()};
            tensor<AccDataType> saveInvVar_ref{derived_layout, derivedBnDesc.GetLengths()};
            tensor<AccDataType> runMean_ref{derived_layout, derivedBnDesc.GetLengths()};
            tensor<AccDataType> runVar_ref{derived_layout, derivedBnDesc.GetLengths()};

            struct
            {
                tensor<T> input;
                tensor<AccDataType> out_ref;
                tensor<AccDataType> scale;
                tensor<AccDataType> shift;
                double epsilon;
                double averageFactor;
                tensor<AccDataType> saveMean_ref;
                tensor<AccDataType> saveVariance_ref;
                tensor<AccDataType> runMean_ref;
                tensor<AccDataType> runVariance_ref;
                miopenBatchNormMode_t bn_mode = miopenBNPerActivation;
            } dl{input,
                 out_ref,
                 scale,
                 shift,
                 epsilon,
                 expAvgFactor,
                 saveMean_ref,
                 saveInvVar_ref,
                 runMean_ref,
                 runVar_ref};

            EnsureValidLayout(dl.input, miopenTensorNCDHW);
            EnsureValidLayout(dl.out_ref, bn_layout);
            EnsureValidLayout(dl.scale, derived_layout);
            EnsureValidLayout(dl.shift, derived_layout);
            EnsureValidLayout(dl.saveMean_ref, derived_layout);
            EnsureValidLayout(dl.saveVariance_ref, derived_layout);
            EnsureValidLayout(dl.runMean_ref, derived_layout);
            EnsureValidLayout(dl.runVariance_ref, derived_layout);

            test::ComputeCPUBNFwdTrain<T, AccDataType>(dl);
            test::CompareTensor(output, out_ref, tolerance);
            test::CompareTensor(saveMean, saveMean_ref, tolerance);
            test::CompareTensor(saveInvVar, saveInvVar_ref, tolerance);
            test::CompareTensor(runMean, runMean_ref, tolerance);
            test::CompareTensor(runVar, runVar_ref, tolerance);
            break;
        }
        case BN3DPerActTestType::ForwardInferenceRecalc:
        case BN3DPerActTestType::ForwardInferenceUseEstimated: {
            void* p_est_mean =
                (test_case.test_type == BN3DPerActTestType::ForwardInferenceUseEstimated)
                    ? runMean_dev.get()
                    : nullptr;
            void* p_est_var =
                (test_case.test_type == BN3DPerActTestType::ForwardInferenceUseEstimated)
                    ? runVar_dev.get()
                    : nullptr;
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
                                                                             p_est_mean,
                                                                             p_est_var,
                                                                             epsilon);

            ASSERT_EQ(status, miopenStatusSuccess);

            output.data = handle.Read<T>(out_dev, output.data.size());

            struct
            {
                tensor<T> input;
                tensor<AccDataType> out_ref;
                tensor<AccDataType> scale;
                tensor<AccDataType> shift;
                tensor<AccDataType> estMean;
                tensor<AccDataType> estVariance;
                double epsilon;
                bool useInverseVariance       = false;
                miopenBatchNormMode_t bn_mode = miopenBNPerActivation;
            } dl{input,
                 out_ref,
                 scale,
                 shift,
                 runMean,
                 runVar,
                 epsilon};

            EnsureValidLayout(dl.input, miopenTensorNCDHW);
            EnsureValidLayout(dl.out_ref, bn_layout);
            EnsureValidLayout(dl.scale, derived_layout);
            EnsureValidLayout(dl.shift, derived_layout);
            EnsureValidLayout(dl.estMean, derived_layout);
            EnsureValidLayout(dl.estVariance, derived_layout);

            if(test_case.test_type == BN3DPerActTestType::ForwardInferenceRecalc)
            {
                struct
                {
                    tensor<T> input;
                    tensor<AccDataType> out_ref;
                    tensor<AccDataType> scale;
                    tensor<AccDataType> shift;
                    double epsilon;
                    double averageFactor;
                    tensor<AccDataType> saveMean_ref;
                    tensor<AccDataType> saveVariance_ref;
                    tensor<AccDataType> runMean_ref;
                    tensor<AccDataType> runVariance_ref;
                    miopenBatchNormMode_t bn_mode = miopenBNPerActivation;
                } dl_fwd{input,
                         out_ref,
                         scale,
                         shift,
                         epsilon,
                         expAvgFactor,
                         runMean,
                         runVar,
                         runMean,
                         runVar};

                EnsureValidLayout(dl_fwd.input, miopenTensorNCDHW);
                EnsureValidLayout(dl_fwd.out_ref, bn_layout);
                test::ComputeCPUBNFwdTrain<T, AccDataType>(dl_fwd);
                dl.estMean.desc       = dl_fwd.saveMean_ref.desc;
                dl.estVariance.desc   = dl_fwd.saveVariance_ref.desc;
                dl.useInverseVariance = true;
            }
            test::ComputeCPUBNInference<T, AccDataType>(dl);
            test::CompareTensor(output, out_ref, tolerance);
            break;
        }
        case BN3DPerActTestType::BackwardRecalc: {
            tensor<T> dy_input{miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w}};
            dy_input.generate(uniform_signed_initializer<T>(2e-3, 1000));
            auto dy_dev = handle.Write(dy_input.data);

            tensor<T> dx_output{miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w}};
            tensor<AccDataType> dscale{derived_layout, derivedBnDesc.GetLengths()};
            tensor<AccDataType> dshift{derived_layout, derivedBnDesc.GetLengths()};
            auto dx_dev     = handle.Write(dx_output.data);
            auto dscale_dev = handle.Write(dscale.data);
            auto dshift_dev = handle.Write(dshift.data);

            miopenStatus_t status = miopenBatchNormalizationBackward(&handle,
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
                                                                     nullptr,
                                                                     nullptr);

            ASSERT_EQ(status, miopenStatusSuccess);

            dx_output.data = handle.Read<T>(dx_dev, dx_output.data.size());
            dscale.data    = handle.Read<AccDataType>(dscale_dev, dscale.data.size());
            dshift.data    = handle.Read<AccDataType>(dshift_dev, dshift.data.size());

            // Create separate tensors for CPU reference results
            tensor<T> dx_output_ref{miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w}};
            tensor<AccDataType> dscale_ref{derived_layout, derivedBnDesc.GetLengths()};
            tensor<AccDataType> dshift_ref{derived_layout, derivedBnDesc.GetLengths()};

            struct
            {
                tensor<T> input;
                tensor<T> dy;
                tensor<T> out_ref;
                tensor<AccDataType> bnScale;
                tensor<AccDataType> bnBias;
                tensor<AccDataType> dScale_ref;
                tensor<AccDataType> dBias_ref;
                tensor<AccDataType> savedMean;
                tensor<AccDataType> savedInvVar;
                double epsilon;
                miopenBatchNormMode_t bn_mode     = miopenBNPerActivation;
                miopenActivationMode_t activ_mode = miopenActivationPASTHRU;
                double activ_alpha                = 1.0;
                double activ_beta                 = 0.0;
            } dl{input,
                 dy_input,
                 dx_output_ref,
                 scale,
                 shift,
                 dscale_ref,
                 dshift_ref,
                 runMean,
                 runVar,
                 epsilon};

            EnsureValidLayout(dl.input, miopenTensorNCDHW);
            EnsureValidLayout(dl.dy, miopenTensorNCDHW);
            EnsureValidLayout(dl.out_ref, miopenTensorNCDHW);
            EnsureValidLayout(dl.bnScale, derived_layout);
            EnsureValidLayout(dl.bnBias, derived_layout);
            EnsureValidLayout(dl.dScale_ref, derived_layout);
            EnsureValidLayout(dl.dBias_ref, derived_layout);

            struct
            {
                tensor<T> input;
                tensor<AccDataType> out_ref;
                tensor<AccDataType> scale;
                tensor<AccDataType> shift;
                double epsilon;
                double averageFactor;
                tensor<AccDataType> saveMean_ref;
                tensor<AccDataType> saveVariance_ref;
                tensor<AccDataType> runMean_ref;
                tensor<AccDataType> runVariance_ref;
                miopenBatchNormMode_t bn_mode = miopenBNPerActivation;
            } dl_fwd{input,
                     out_ref,
                     scale,
                     shift,
                     epsilon,
                     expAvgFactor,
                     runMean,
                     runVar,
                     runMean,
                     runVar};

            EnsureValidLayout(dl_fwd.input, miopenTensorNCDHW);
            EnsureValidLayout(dl_fwd.out_ref, bn_layout);
            EnsureValidLayout(dl_fwd.scale, derived_layout);
            EnsureValidLayout(dl_fwd.shift, derived_layout);
            EnsureValidLayout(dl_fwd.saveMean_ref, derived_layout);
            EnsureValidLayout(dl_fwd.saveVariance_ref, derived_layout);
            EnsureValidLayout(dl_fwd.runMean_ref, derived_layout);
            EnsureValidLayout(dl_fwd.runVariance_ref, derived_layout);

            test::ComputeCPUBNFwdTrain<T, AccDataType>(dl_fwd);
            dl.savedMean.desc   = dl_fwd.saveMean_ref.desc;
            dl.savedInvVar.desc = dl_fwd.saveVariance_ref.desc;
            EnsureValidLayout(dl.savedMean, derived_layout);
            EnsureValidLayout(dl.savedInvVar, derived_layout);

            test::ComputeCPUBNBwd<T, AccDataType>(dl);
            test::CompareTensor(dx_output, dx_output_ref, tolerance);
            test::CompareTensor(dscale, dscale_ref, tolerance);
            test::CompareTensor(dshift, dshift_ref, tolerance);
            break;
        }
        case BN3DPerActTestType::BackwardUseSaved: {
            tensor<AccDataType> saveMean{derived_layout, derivedBnDesc.GetLengths()};
            tensor<AccDataType> saveInvVar{derived_layout, derivedBnDesc.GetLengths()};
            auto saveMean_dev   = handle.Write(saveMean.data);
            auto saveInvVar_dev = handle.Write(saveInvVar.data);

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
                                                                            saveMean_dev.get(),
                                                                            saveInvVar_dev.get());

            ASSERT_EQ(status, miopenStatusSuccess);

            saveMean.data   = handle.Read<AccDataType>(saveMean_dev, saveMean.data.size());
            saveInvVar.data = handle.Read<AccDataType>(saveInvVar_dev, saveInvVar.data.size());

            tensor<T> dy_input{miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w}};
            dy_input.generate(uniform_signed_initializer<T>(2e-3, 1000));
            auto dy_dev = handle.Write(dy_input.data);

            tensor<T> dx_output{miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w}};
            tensor<AccDataType> dscale{derived_layout, derivedBnDesc.GetLengths()};
            tensor<AccDataType> dshift{derived_layout, derivedBnDesc.GetLengths()};
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
                                                      saveMean_dev.get(),
                                                      saveInvVar_dev.get());

            ASSERT_EQ(status, miopenStatusSuccess);

            dx_output.data = handle.Read<T>(dx_dev, dx_output.data.size());
            dscale.data    = handle.Read<AccDataType>(dscale_dev, dscale.data.size());
            dshift.data    = handle.Read<AccDataType>(dshift_dev, dshift.data.size());

            // Create separate tensors for CPU reference results
            tensor<T> dx_output_ref{miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w}};
            tensor<AccDataType> dscale_ref{derived_layout, derivedBnDesc.GetLengths()};
            tensor<AccDataType> dshift_ref{derived_layout, derivedBnDesc.GetLengths()};

            struct
            {
                tensor<T> input;
                tensor<T> dy;
                tensor<T> out_ref;
                tensor<AccDataType> bnScale;
                tensor<AccDataType> bnBias;
                tensor<AccDataType> dScale_ref;
                tensor<AccDataType> dBias_ref;
                tensor<AccDataType> savedMean;
                tensor<AccDataType> savedInvVar;
                double epsilon;
                miopenBatchNormMode_t bn_mode     = miopenBNPerActivation;
                miopenActivationMode_t activ_mode = miopenActivationPASTHRU;
                double activ_alpha                = 1.0;
                double activ_beta                 = 0.0;
            } dl{input,
                 dy_input,
                 dx_output_ref,
                 scale,
                 shift,
                 dscale_ref,
                 dshift_ref,
                 saveMean,
                 saveInvVar,
                 epsilon};

            EnsureValidLayout(dl.input, miopenTensorNCDHW);
            EnsureValidLayout(dl.dy, miopenTensorNCDHW);
            EnsureValidLayout(dl.out_ref, miopenTensorNCDHW);
            EnsureValidLayout(dl.bnScale, derived_layout);
            EnsureValidLayout(dl.bnBias, derived_layout);
            EnsureValidLayout(dl.dScale_ref, derived_layout);
            EnsureValidLayout(dl.dBias_ref, derived_layout);
            EnsureValidLayout(dl.savedMean, derived_layout);
            EnsureValidLayout(dl.savedInvVar, derived_layout);

            test::ComputeCPUBNBwd<T, AccDataType>(dl);
            test::CompareTensor(dx_output, dx_output_ref, tolerance);
            test::CompareTensor(dscale, dscale_ref, tolerance);
            test::CompareTensor(dshift, dshift_ref, tolerance);
            break;
        }
        }
    }
};

template <typename T>
tensor<T> GPU_Bn3dPerAct<T>::pool_input;
template <typename T>
tensor<typename GPU_Bn3dPerAct<T>::AccDataType> GPU_Bn3dPerAct<T>::pool_scale;
template <typename T>
tensor<typename GPU_Bn3dPerAct<T>::AccDataType> GPU_Bn3dPerAct<T>::pool_shift;
template <typename T>
tensor<typename GPU_Bn3dPerAct<T>::AccDataType> GPU_Bn3dPerAct<T>::pool_runMean;
template <typename T>
tensor<typename GPU_Bn3dPerAct<T>::AccDataType> GPU_Bn3dPerAct<T>::pool_runVar;

using GPU_Bn3dPerAct_FP32  = GPU_Bn3dPerAct<float>;
using GPU_Bn3dPerAct_FP16  = GPU_Bn3dPerAct<half_float::half>;
using GPU_Bn3dPerAct_BFP16 = GPU_Bn3dPerAct<bfloat16>;
using GPU_Bn3dPerAct_FP64  = GPU_Bn3dPerAct<double>;

} // namespace

TEST_P(GPU_Bn3dPerAct_FP32, Test) { this->RunTest(); }
TEST_P(GPU_Bn3dPerAct_FP16, Test) { this->RunTest(); }
TEST_P(GPU_Bn3dPerAct_BFP16, Test) { this->RunTest(); }

// Match ctest: only run FP32, FP16, and BF16 (like 2D BN peract test)
INSTANTIATE_TEST_SUITE_P(Full, GPU_Bn3dPerAct_FP32, testing::ValuesIn(GetBN3DPerActTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, GPU_Bn3dPerAct_FP16, testing::ValuesIn(GetBN3DPerActTestCases()));

INSTANTIATE_TEST_SUITE_P(Full, GPU_Bn3dPerAct_BFP16, testing::ValuesIn(GetBN3DPerActTestCases()));
