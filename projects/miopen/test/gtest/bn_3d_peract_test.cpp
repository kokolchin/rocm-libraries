// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/batch_norm.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <utility>

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

template <typename T, typename AccDataType>
struct DLModuleHelper
{
    struct DLModule
    {
        tensor<T> input;
        tensor<T> output;
        tensor<AccDataType> out_ref;
        tensor<AccDataType> scale;
        tensor<AccDataType> shift;
        tensor<AccDataType> estMean;
        tensor<AccDataType> estVariance;
        tensor<T> dy;
        tensor<AccDataType> bnScale;
        tensor<AccDataType> bnBias;
        tensor<AccDataType> dScale_ref;
        tensor<AccDataType> dBias_ref;
        tensor<AccDataType> savedMean;
        tensor<AccDataType> savedInvVar;
        tensor<AccDataType> saveMean_ref;
        tensor<AccDataType> saveVariance_ref;
        tensor<AccDataType> runMean_ref;
        tensor<AccDataType> runVariance_ref;

        miopenBatchNormMode_t bn_mode     = miopenBNPerActivation;
        double epsilon                    = MIO_BN_TEST_EPSILON;
        double averageFactor              = MIO_BN_TEST_EXPAVGFACTOR;
        bool useInverseVariance           = false;
        miopenActivationMode_t activ_mode = miopenActivationPASTHRU;
        double activ_alpha                = 1.0;
        double activ_beta                 = 0.0;
        double activ_gamma                = 1.0;
    };

    template <typename U>
    static void MoveTo(tensor<U>& src, tensor<U>& dst)
    {
        dst.desc = src.desc;
        dst.data = std::move(src.data);
    }
    template <typename U>
    static void MoveBack(tensor<U>& src, tensor<U>& dst)
    {
        dst.data = std::move(src.data);
    }
};

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

enum class BN3DPerActTestSet
{
    Standard, // 4 types: FwdTrain, FwdInferenceRecalc, BwdRecalc, BwdUseSaved
    Full      // 5 types: includes ForwardInferenceUseEstimated
};

std::vector<BN3DPerActTestCase>
GetBN3DPerActTestCases(BN3DPerActTestSet test_set = BN3DPerActTestSet::Standard)
{
    std::vector<BN3DPerActTestCase> test_cases;
    // Match ctest behavior:
    // FP32 runs 5 types (Full), while FP16/BF16 run 4 types (Standard)
    std::vector<BN3DPerActTestType> types;
    if(test_set == BN3DPerActTestSet::Full)
    {
        types = {BN3DPerActTestType::ForwardTraining,
                 BN3DPerActTestType::ForwardInferenceRecalc,
                 BN3DPerActTestType::ForwardInferenceUseEstimated,
                 BN3DPerActTestType::BackwardRecalc,
                 BN3DPerActTestType::BackwardUseSaved};
    }
    else
    {
        types = {BN3DPerActTestType::ForwardTraining,
                 BN3DPerActTestType::ForwardInferenceRecalc,
                 BN3DPerActTestType::BackwardRecalc,
                 BN3DPerActTestType::BackwardUseSaved};
    }

    // Use batch size factor 4 to match ctest behavior (like other BN 3D tests)
    for(const auto& shape : get_3d_bn_peract_inputs(4))
    {
        const auto n = shape[0];
        // Match ctest logic: skip ALL test cases when n == 1
        if(n == 1)
        {
            continue;
        }
        for(const auto& type : types)
        {
            test_cases.push_back({shape[0], shape[1], shape[2], shape[3], shape[4], type});
        }
    }
    return test_cases;
}

template <typename T>
struct GPU_Bn3dPerAct : public ::testing::TestWithParam<BN3DPerActTestCase>
{
    using AccDataType = std::conditional_t<std::is_same_v<T, double>, double, float>;

    struct Cache
    {
        std::vector<std::size_t> shape;
        tensor<T> input;
        tensor<AccDataType> scale;
        tensor<AccDataType> shift;
        miopen::Allocator::ManageDataPtr in_dev;
        miopen::Allocator::ManageDataPtr scale_dev;
        miopen::Allocator::ManageDataPtr shift_dev;
    };

    static Cache& GetCache()
    {
        static Cache cache;
        return cache;
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

        if(n == 1)
        {
            GTEST_SKIP() << "Invalid batch size for batch norm tests";
        }

        auto&& handle = get_handle();

        std::vector<std::size_t> current_shape = {n, c, d, h, w};
        auto& cache                            = GetCache();

        if(cache.shape == current_shape && !cache.input.data.empty())
        {
            // Reuse cached read-only tensors and GPU memory
            input     = std::move(cache.input);
            scale     = std::move(cache.scale);
            shift     = std::move(cache.shift);
            in_dev    = std::move(cache.in_dev);
            scale_dev = std::move(cache.scale_dev);
            shift_dev = std::move(cache.shift_dev);

            miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNPerActivation);
        }
        else
        {
            // New shape or empty cache, initialize everything
            input = tensor<T>{miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w}};
            input.generate(uniform_signed_initializer<T>(2e-3, 1000));
            in_dev = handle.Write(input.data);

            miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNPerActivation);

            auto dims_num  = derivedBnDesc.GetLengths().size();
            derived_layout = (dims_num == 5) ? miopenTensorNCDHW : miopenTensorNCHW;
            derivedBnDesc  = miopen::TensorDescriptor(
                derivedBnDesc.GetType(), derived_layout, derivedBnDesc.GetLengths());

            scale = tensor<AccDataType>{derived_layout, derivedBnDesc.GetLengths()};
            shift = tensor<AccDataType>{derived_layout, derivedBnDesc.GetLengths()};

            scale.generate(uniform_signed_initializer<AccDataType>(2e-3, 1000));
            shift.generate(uniform_signed_initializer<AccDataType>(2e-3, 1000));

            scale_dev = handle.Write(scale.data);
            shift_dev = handle.Write(shift.data);
        }

        // Re-generate runMean and runVar every time to ensure test independence
        auto dims_num  = derivedBnDesc.GetLengths().size();
        derived_layout = (dims_num == 5) ? miopenTensorNCDHW : miopenTensorNCHW;
        runMean        = tensor<AccDataType>{derived_layout, derivedBnDesc.GetLengths()};
        runVar         = tensor<AccDataType>{derived_layout, derivedBnDesc.GetLengths()};
        runMean.generate(uniform_signed_initializer<AccDataType>(2e-3, 1000));
        runVar.generate(uniform_unsigned_initializer<AccDataType>(2e-3, 1000));
        runMean_dev = handle.Write(runMean.data);
        runVar_dev  = handle.Write(runVar.data);

        output  = tensor<T>{miopenTensorNCDHW, std::vector<std::size_t>{n, c, d, h, w}};
        out_dev = handle.Write(output.data);

        auto input_layout_opt = input.desc.GetLayoutEnum();
        bn_layout = (input_layout_opt && input_layout_opt.value() != 0) ? input_layout_opt.value()
                                                                        : miopenTensorNCDHW;
        if(bn_layout == 0)
        {
            bn_layout = miopenTensorNCDHW;
        }
        out_ref   = tensor<AccDataType>{bn_layout, std::vector<std::size_t>{n, c, d, h, w}};
        tolerance = 0.5;
    }

    void TearDown() override
    {
        if(!input.data.empty())
        {
            auto& cache     = GetCache();
            cache.shape     = {n, c, d, h, w};
            cache.input     = std::move(input);
            cache.scale     = std::move(scale);
            cache.shift     = std::move(shift);
            cache.in_dev    = std::move(in_dev);
            cache.scale_dev = std::move(cache.scale_dev);
            cache.shift_dev = std::move(cache.shift_dev);
        }
    }

    template <typename TensorType>
    static void EnsureValidLayout(TensorType& t, miopenTensorLayout_t default_layout)
    {
        auto layout_t = t.desc.GetLayout_t();
        auto num_dims = t.desc.GetLengths().size();

        miopenTensorLayout_t valid_layout;
        if(num_dims == 5)
        {
            valid_layout = miopenTensorNCDHW;
        }
        else if(num_dims == 4)
        {
            valid_layout = miopenTensorNCHW;
        }
        else
        {
            valid_layout = default_layout;
        }

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

    using DLModule = typename DLModuleHelper<T, AccDataType>::DLModule;

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
    double tolerance    = 5e-3;
};

using GPU_Bn3dPerAct_FP32  = GPU_Bn3dPerAct<float>;
using GPU_Bn3dPerAct_FP16  = GPU_Bn3dPerAct<half_float::half>;
using GPU_Bn3dPerAct_BFP16 = GPU_Bn3dPerAct<bfloat16>;

#define TEST_PERACT_3D(fixture, data_type)                                                         \
    TEST_P(fixture, Test)                                                                          \
    {                                                                                              \
        using AccDataType     = typename fixture::AccDataType;                                     \
        using Helper          = DLModuleHelper<data_type, AccDataType>;                            \
        const auto& test_case = this->GetParam();                                                  \
        auto&& handle         = get_handle();                                                      \
                                                                                                   \
        switch(test_case.test_type)                                                                \
        {                                                                                          \
        case BN3DPerActTestType::ForwardTraining: {                                                \
            tensor<AccDataType> saveMean{this->derived_layout, this->derivedBnDesc.GetLengths()};  \
            tensor<AccDataType> saveInvVar{this->derived_layout,                                   \
                                           this->derivedBnDesc.GetLengths()};                      \
            auto saveMean_dev   = handle.Write(saveMean.data);                                     \
            auto saveInvVar_dev = handle.Write(saveInvVar.data);                                   \
                                                                                                   \
            miopenStatus_t status = miopenBatchNormalizationForwardTraining(&handle,               \
                                                                            miopenBNPerActivation, \
                                                                            &alpha,                \
                                                                            &beta,                 \
                                                                            &input.desc,           \
                                                                            in_dev.get(),          \
                                                                            &output.desc,          \
                                                                            out_dev.get(),         \
                                                                            &derivedBnDesc,        \
                                                                            scale_dev.get(),       \
                                                                            shift_dev.get(),       \
                                                                            expAvgFactor,          \
                                                                            runMean_dev.get(),     \
                                                                            runVar_dev.get(),      \
                                                                            epsilon,               \
                                                                            saveMean_dev.get(),    \
                                                                            saveInvVar_dev.get()); \
                                                                                                   \
            ASSERT_EQ(status, miopenStatusSuccess);                                                \
                                                                                                   \
            /* Read GPU results into separate tensor objects to avoid overwriting initial state */ \
            tensor<data_type> gpu_output(output.desc);                                             \
            gpu_output.data = handle.Read<data_type>(out_dev, output.data.size());                 \
            tensor<AccDataType> gpu_saveMean(saveMean.desc);                                       \
            gpu_saveMean.data = handle.Read<AccDataType>(saveMean_dev, saveMean.data.size());      \
            tensor<AccDataType> gpu_saveInvVar(saveInvVar.desc);                                   \
            gpu_saveInvVar.data = handle.Read<AccDataType>(saveInvVar_dev, saveInvVar.data.size());\
            tensor<AccDataType> gpu_runMean(runMean.desc);                                         \
            gpu_runMean.data = handle.Read<AccDataType>(runMean_dev, runMean.data.size());          \
            tensor<AccDataType> gpu_runVar(runVar.desc);                                           \
            gpu_runVar.data = handle.Read<AccDataType>(runVar_dev, runVar.data.size());            \
                                                                                                   \
            typename fixture::DLModule dl;                                                         \
            Helper::MoveTo(input, dl.input);                                                       \
            Helper::MoveTo(scale, dl.scale);                                                       \
            Helper::MoveTo(shift, dl.shift);                                                       \
            Helper::MoveTo(runMean, dl.runMean_ref);                                               \
            Helper::MoveTo(runVar, dl.runVariance_ref);                                            \
            Helper::MoveTo(out_ref, dl.out_ref);                                                   \
                                                                                                   \
            EnsureValidLayout(dl.input, miopenTensorNCDHW);                                        \
            EnsureValidLayout(dl.out_ref, this->bn_layout);                                        \
            EnsureValidLayout(dl.scale, this->derived_layout);                                     \
            EnsureValidLayout(dl.shift, this->derived_layout);                                     \
            EnsureValidLayout(dl.runMean_ref, this->derived_layout);                               \
            EnsureValidLayout(dl.runVariance_ref, this->derived_layout);                           \
                                                                                                   \
            dl.saveMean_ref     = tensor<AccDataType>(this->derivedBnDesc);                        \
            dl.saveVariance_ref = tensor<AccDataType>(this->derivedBnDesc);                        \
            EnsureValidLayout(dl.saveMean_ref, this->derived_layout);                              \
            EnsureValidLayout(dl.saveVariance_ref, this->derived_layout);                          \
                                                                                                   \
            test::ComputeCPUBNFwdTrain(dl);                                                        \
                                                                                                   \
            test::CompareTensor(gpu_output, dl.out_ref, tolerance);                                 \
            test::CompareTensor(gpu_saveMean, dl.saveMean_ref, tolerance);                         \
            test::CompareTensor(gpu_saveInvVar, dl.saveVariance_ref, tolerance);                   \
            test::CompareTensor(gpu_runMean, dl.runMean_ref, tolerance);                           \
            test::CompareTensor(gpu_runVar, dl.runVariance_ref, tolerance);                        \
                                                                                                   \
            Helper::MoveBack(dl.input, input);                                                     \
            Helper::MoveBack(dl.scale, scale);                                                     \
            Helper::MoveBack(dl.shift, shift);                                                     \
            Helper::MoveBack(dl.out_ref, out_ref);                                                 \
            break;                                                                                 \
        }                                                                                          \
        case BN3DPerActTestType::ForwardInferenceRecalc:                                           \
        case BN3DPerActTestType::ForwardInferenceUseEstimated: {                                   \
            void* p_est_mean =                                                                     \
                (test_case.test_type == BN3DPerActTestType::ForwardInferenceUseEstimated)          \
                    ? runMean_dev.get()                                                            \
                    : nullptr;                                                                     \
            void* p_est_var =                                                                      \
                (test_case.test_type == BN3DPerActTestType::ForwardInferenceUseEstimated)          \
                    ? runVar_dev.get()                                                             \
                    : nullptr;                                                                     \
            miopenStatus_t status =                                                                \
                miopenBatchNormalizationForwardInference(&handle,                                  \
                                                         miopenBNPerActivation,                    \
                                                         &alpha,                                   \
                                                         &beta,                                    \
                                                         &input.desc,                              \
                                                         in_dev.get(),                             \
                                                         &output.desc,                             \
                                                         out_dev.get(),                            \
                                                         &derivedBnDesc,                           \
                                                         scale_dev.get(),                          \
                                                         shift_dev.get(),                          \
                                                         p_est_mean,                               \
                                                         p_est_var,                                \
                                                         epsilon);                                 \
                                                                                                   \
            ASSERT_EQ(status, miopenStatusSuccess);                                                \
                                                                                                   \
            tensor<data_type> gpu_output(output.desc);                                             \
            gpu_output.data = handle.Read<data_type>(out_dev, output.data.size());                 \
                                                                                                   \
            typename fixture::DLModule dl;                                                         \
            Helper::MoveTo(input, dl.input);                                                       \
            Helper::MoveTo(scale, dl.scale);                                                       \
            Helper::MoveTo(shift, dl.shift);                                                       \
            Helper::MoveTo(runMean, dl.estMean);                                                   \
            Helper::MoveTo(runVar, dl.estVariance);                                                \
            Helper::MoveTo(out_ref, dl.out_ref);                                                   \
                                                                                                   \
            dl.useInverseVariance = false;                                                         \
            EnsureValidLayout(dl.input, miopenTensorNCDHW);                                        \
            EnsureValidLayout(dl.out_ref, this->bn_layout);                                        \
            EnsureValidLayout(dl.scale, this->derived_layout);                                     \
            EnsureValidLayout(dl.shift, this->derived_layout);                                     \
            EnsureValidLayout(dl.estMean, this->derived_layout);                                   \
            EnsureValidLayout(dl.estVariance, this->derived_layout);                               \
                                                                                                   \
            if(test_case.test_type == BN3DPerActTestType::ForwardInferenceRecalc)                  \
            {                                                                                      \
                typename fixture::DLModule dl_fwd;                                                 \
                dl_fwd.input.desc            = dl.input.desc;                                      \
                dl_fwd.input.data            = std::move(dl.input.data);                           \
                dl_fwd.out_ref.desc          = dl.out_ref.desc;                                    \
                dl_fwd.out_ref.data          = std::move(dl.out_ref.data);                         \
                dl_fwd.scale.desc            = dl.scale.desc;                                      \
                dl_fwd.scale.data            = std::move(dl.scale.data);                           \
                dl_fwd.shift.desc            = dl.shift.desc;                                      \
                dl_fwd.shift.data            = std::move(dl.shift.data);                           \
                dl_fwd.saveMean_ref.desc     = dl.estMean.desc;                                    \
                dl_fwd.saveMean_ref.data     = std::move(dl.estMean.data);                         \
                dl_fwd.saveVariance_ref.desc = dl.estVariance.desc;                                \
                dl_fwd.saveVariance_ref.data = std::move(dl.estVariance.data);                     \
                dl_fwd.runMean_ref.desc      = dl_fwd.saveMean_ref.desc;                           \
                dl_fwd.runVariance_ref.desc  = dl_fwd.saveVariance_ref.desc;                       \
                EnsureValidLayout(dl_fwd.input, miopenTensorNCDHW);                                \
                EnsureValidLayout(dl_fwd.out_ref, this->bn_layout);                                \
                test::ComputeCPUBNFwdTrain(dl_fwd);                                                \
                dl.input.data         = std::move(dl_fwd.input.data);                              \
                dl.out_ref.data       = std::move(dl_fwd.out_ref.data);                            \
                dl.scale.data         = std::move(dl_fwd.scale.data);                              \
                dl.shift.data         = std::move(dl_fwd.shift.data);                              \
                dl.estMean.data       = std::move(dl_fwd.saveMean_ref.data);                       \
                dl.estVariance.data   = std::move(dl_fwd.saveVariance_ref.data);                   \
                dl.useInverseVariance = true;                                                      \
            }                                                                                      \
            test::ComputeCPUBNInference(dl);                                                       \
            test::CompareTensor(gpu_output, dl.out_ref, tolerance);                                \
                                                                                                   \
            Helper::MoveBack(dl.input, input);                                                     \
            Helper::MoveBack(dl.scale, scale);                                                     \
            Helper::MoveBack(dl.shift, shift);                                                     \
            Helper::MoveBack(dl.out_ref, out_ref);                                                 \
            Helper::MoveBack(dl.estMean, runMean);                                                 \
            Helper::MoveBack(dl.estVariance, runVar);                                              \
            break;                                                                                 \
        }                                                                                          \
        case BN3DPerActTestType::BackwardRecalc: {                                                 \
            tensor<data_type> dy_input{miopenTensorNCDHW,                                          \
                                       std::vector<std::size_t>{n, c, d, h, w}};                   \
            dy_input.generate(uniform_signed_initializer<data_type>(2e-3, 1000));                  \
            auto dy_dev = handle.Write(dy_input.data);                                             \
                                                                                                   \
            tensor<data_type> dx_output{miopenTensorNCDHW,                                         \
                                        std::vector<std::size_t>{n, c, d, h, w}};                  \
            tensor<AccDataType> dscale{this->derived_layout, this->derivedBnDesc.GetLengths()};    \
            tensor<AccDataType> dshift{this->derived_layout, this->derivedBnDesc.GetLengths()};    \
            auto dx_dev     = handle.Write(dx_output.data);                                        \
            auto dscale_dev = handle.Write(dscale.data);                                           \
            auto dshift_dev = handle.Write(dshift.data);                                           \
                                                                                                   \
            miopenStatus_t status = miopenBatchNormalizationBackward(&handle,                      \
                                                                     miopenBNPerActivation,        \
                                                                     &alpha,                       \
                                                                     &beta,                        \
                                                                     &alpha,                       \
                                                                     &beta,                        \
                                                                     &input.desc,                  \
                                                                     in_dev.get(),                 \
                                                                     &dy_input.desc,               \
                                                                     dy_dev.get(),                 \
                                                                     &dx_output.desc,              \
                                                                     dx_dev.get(),                 \
                                                                     &derivedBnDesc,               \
                                                                     scale_dev.get(),              \
                                                                     dscale_dev.get(),             \
                                                                     dshift_dev.get(),             \
                                                                     epsilon,                      \
                                                                     nullptr,                      \
                                                                     nullptr);                     \
                                                                                                   \
            ASSERT_EQ(status, miopenStatusSuccess);                                                \
                                                                                                   \
            tensor<data_type> gpu_dx(dx_output.desc);                                              \
            gpu_dx.data = handle.Read<data_type>(dx_dev, dx_output.data.size());                   \
            tensor<AccDataType> gpu_dscale(dscale.desc);                                           \
            gpu_dscale.data = handle.Read<AccDataType>(dscale_dev, dscale.data.size());             \
            tensor<AccDataType> gpu_dshift(dshift.desc);                                           \
            gpu_dshift.data = handle.Read<AccDataType>(dshift_dev, dshift.data.size());             \
                                                                                                   \
            typename fixture::DLModule dl;                                                         \
            Helper::MoveTo(input, dl.input);                                                       \
            Helper::MoveTo(dy_input, dl.dy);                                                       \
            Helper::MoveTo(scale, dl.bnScale);                                                     \
            Helper::MoveTo(out_ref, dl.out_ref);                                                   \
                                                                                                   \
            typename fixture::DLModule dl_fwd;                                                     \
            dl_fwd.input.desc   = dl.input.desc;                                                   \
            dl_fwd.input.data   = std::move(dl.input.data);                                        \
            dl_fwd.scale.desc   = dl.bnScale.desc;                                                 \
            dl_fwd.scale.data   = std::move(dl.bnScale.data);                                      \
            dl_fwd.shift.desc   = shift.desc;                                                      \
            dl_fwd.shift.data   = std::move(shift.data);                                           \
            dl_fwd.out_ref.desc = dl.out_ref.desc;                                                 \
            dl_fwd.out_ref.data = std::move(dl.out_ref.data);                                      \
                                                                                                   \
            dl_fwd.saveMean_ref     = tensor<AccDataType>(this->derivedBnDesc);                    \
            dl_fwd.saveVariance_ref = tensor<AccDataType>(this->derivedBnDesc);                    \
            dl_fwd.runMean_ref.desc     = runMean.desc;                                            \
            dl_fwd.runMean_ref.data     = std::move(runMean.data);                                 \
            dl_fwd.runVariance_ref.desc = runVar.desc;                                             \
            dl_fwd.runVariance_ref.data = std::move(runVar.data);                                  \
                                                                                                   \
            EnsureValidLayout(dl_fwd.input, miopenTensorNCDHW);                                    \
            EnsureValidLayout(dl_fwd.scale, this->derived_layout);                                 \
            EnsureValidLayout(dl_fwd.shift, this->derived_layout);                                 \
            EnsureValidLayout(dl_fwd.out_ref, this->bn_layout);                                    \
            EnsureValidLayout(dl_fwd.saveMean_ref, this->derived_layout);                          \
            EnsureValidLayout(dl_fwd.saveVariance_ref, this->derived_layout);                      \
                                                                                                   \
            test::ComputeCPUBNFwdTrain(dl_fwd);                                                    \
                                                                                                   \
            dl.input.data       = std::move(dl_fwd.input.data);                                    \
            dl.bnScale.data     = std::move(dl_fwd.scale.data);                                    \
            dl.out_ref.data     = std::move(dl_fwd.out_ref.data);                                  \
            dl.savedMean        = std::move(dl_fwd.saveMean_ref);                                  \
            dl.savedInvVar      = std::move(dl_fwd.saveVariance_ref);                              \
                                                                                                   \
            EnsureValidLayout(dl.input, miopenTensorNCDHW);                                        \
            EnsureValidLayout(dl.dy, miopenTensorNCDHW);                                           \
            EnsureValidLayout(dl.bnScale, this->derived_layout);                                   \
            EnsureValidLayout(dl.out_ref, this->bn_layout);                                        \
            EnsureValidLayout(dl.savedMean, this->derived_layout);                                 \
            EnsureValidLayout(dl.savedInvVar, this->derived_layout);                               \
                                                                                                   \
            dl.dScale_ref = tensor<AccDataType>(this->derivedBnDesc);                              \
            dl.dBias_ref  = tensor<AccDataType>(this->derivedBnDesc);                              \
            EnsureValidLayout(dl.dScale_ref, this->derived_layout);                                \
            EnsureValidLayout(dl.dBias_ref, this->derived_layout);                                 \
                                                                                                   \
            test::ComputeCPUBNBwd(dl);                                                             \
                                                                                                   \
            test::CompareTensor(gpu_dx, dl.out_ref, tolerance);                                    \
            test::CompareTensor(gpu_dscale, dl.dScale_ref, tolerance);                             \
            test::CompareTensor(gpu_dshift, dl.dBias_ref, tolerance);                              \
                                                                                                   \
            Helper::MoveBack(dl.input, input);                                                     \
            Helper::MoveBack(dl.dy, dy_input);                                                     \
            Helper::MoveBack(dl.bnScale, scale);                                                   \
            Helper::MoveBack(dl.out_ref, out_ref);                                                 \
            shift.data   = std::move(dl_fwd.shift.data);                                           \
            runMean.data = std::move(dl_fwd.runMean_ref.data);                                     \
            runVar.data  = std::move(dl_fwd.runVariance_ref.data);                                 \
            break;                                                                                 \
        }                                                                                          \
        case BN3DPerActTestType::BackwardUseSaved: {                                               \
            tensor<AccDataType> saveMean{this->derived_layout, this->derivedBnDesc.GetLengths()};  \
            tensor<AccDataType> saveInvVar{this->derived_layout,                                   \
                                           this->derivedBnDesc.GetLengths()};                      \
            auto saveMean_dev   = handle.Write(saveMean.data);                                     \
            auto saveInvVar_dev = handle.Write(saveInvVar.data);                                   \
                                                                                                   \
            miopenStatus_t status = miopenBatchNormalizationForwardTraining(&handle,               \
                                                                            miopenBNPerActivation, \
                                                                            &alpha,                \
                                                                            &beta,                 \
                                                                            &input.desc,           \
                                                                            in_dev.get(),          \
                                                                            &output.desc,          \
                                                                            out_dev.get(),         \
                                                                            &derivedBnDesc,        \
                                                                            scale_dev.get(),       \
                                                                            shift_dev.get(),       \
                                                                            expAvgFactor,          \
                                                                            runMean_dev.get(),     \
                                                                            runVar_dev.get(),      \
                                                                            epsilon,               \
                                                                            saveMean_dev.get(),    \
                                                                            saveInvVar_dev.get()); \
                                                                                                   \
            ASSERT_EQ(status, miopenStatusSuccess);                                                \
                                                                                                   \
            tensor<AccDataType> gpu_saveMean(saveMean.desc);                                       \
            gpu_saveMean.data = handle.Read<AccDataType>(saveMean_dev, saveMean.data.size());      \
            tensor<AccDataType> gpu_saveInvVar(saveInvVar.desc);                                   \
            gpu_saveInvVar.data = handle.Read<AccDataType>(saveInvVar_dev, saveInvVar.data.size());\
                                                                                                   \
            tensor<data_type> dy_input{miopenTensorNCDHW,                                          \
                                       std::vector<std::size_t>{n, c, d, h, w}};                   \
            dy_input.generate(uniform_signed_initializer<data_type>(2e-3, 1000));                  \
            auto dy_dev = handle.Write(dy_input.data);                                             \
                                                                                                   \
            tensor<data_type> dx_output{miopenTensorNCDHW,                                         \
                                        std::vector<std::size_t>{n, c, d, h, w}};                  \
            tensor<AccDataType> dscale{this->derived_layout, this->derivedBnDesc.GetLengths()};    \
            tensor<AccDataType> dshift{this->derived_layout, this->derivedBnDesc.GetLengths()};    \
            auto dx_dev     = handle.Write(dx_output.data);                                        \
            auto dscale_dev = handle.Write(dscale.data);                                           \
            auto dshift_dev = handle.Write(dshift.data);                                           \
                                                                                                   \
            status = miopenBatchNormalizationBackward(&handle,                                     \
                                                      miopenBNPerActivation,                       \
                                                      &alpha,                                      \
                                                      &beta,                                       \
                                                      &alpha,                                      \
                                                      &beta,                                       \
                                                      &input.desc,                                 \
                                                      in_dev.get(),                                \
                                                      &dy_input.desc,                              \
                                                      dy_dev.get(),                                \
                                                      &dx_output.desc,                             \
                                                      dx_dev.get(),                                \
                                                      &derivedBnDesc,                              \
                                                      scale_dev.get(),                             \
                                                      dscale_dev.get(),                            \
                                                      dshift_dev.get(),                            \
                                                      epsilon,                                     \
                                                      saveMean_dev.get(),                          \
                                                      saveInvVar_dev.get());                       \
                                                                                                   \
            ASSERT_EQ(status, miopenStatusSuccess);                                                \
                                                                                                   \
            tensor<data_type> gpu_dx(dx_output.desc);                                              \
            gpu_dx.data = handle.Read<data_type>(dx_dev, dx_output.data.size());                   \
            tensor<AccDataType> gpu_dscale(dscale.desc);                                           \
            gpu_dscale.data = handle.Read<AccDataType>(dscale_dev, dscale.data.size());             \
            tensor<AccDataType> gpu_dshift(dshift.desc);                                           \
            gpu_dshift.data = handle.Read<AccDataType>(dshift_dev, dshift.data.size());             \
                                                                                                   \
            typename fixture::DLModule dl;                                                         \
            Helper::MoveTo(input, dl.input);                                                       \
            Helper::MoveTo(dy_input, dl.dy);                                                       \
            Helper::MoveTo(scale, dl.bnScale);                                                     \
            Helper::MoveTo(out_ref, dl.out_ref);                                                   \
            Helper::MoveTo(gpu_saveMean, dl.savedMean);                                            \
            Helper::MoveTo(gpu_saveInvVar, dl.savedInvVar);                                        \
                                                                                                   \
            EnsureValidLayout(dl.input, miopenTensorNCDHW);                                        \
            EnsureValidLayout(dl.dy, miopenTensorNCDHW);                                           \
            EnsureValidLayout(dl.bnScale, this->derived_layout);                                   \
            EnsureValidLayout(dl.out_ref, this->bn_layout);                                        \
            EnsureValidLayout(dl.savedMean, this->derived_layout);                                 \
            EnsureValidLayout(dl.savedInvVar, this->derived_layout);                               \
                                                                                                   \
            dl.dScale_ref = tensor<AccDataType>(this->derivedBnDesc);                              \
            dl.dBias_ref  = tensor<AccDataType>(this->derivedBnDesc);                              \
            EnsureValidLayout(dl.dScale_ref, this->derived_layout);                                \
            EnsureValidLayout(dl.dBias_ref, this->derived_layout);                                 \
                                                                                                   \
            test::ComputeCPUBNBwd(dl);                                                             \
                                                                                                   \
            test::CompareTensor(gpu_dx, dl.out_ref, tolerance);                                    \
            test::CompareTensor(gpu_dscale, dl.dScale_ref, tolerance);                             \
            test::CompareTensor(gpu_dshift, dl.dBias_ref, tolerance);                              \
                                                                                                   \
            Helper::MoveBack(dl.input, input);                                                     \
            Helper::MoveBack(dl.dy, dy_input);                                                     \
            Helper::MoveBack(dl.bnScale, scale);                                                   \
            Helper::MoveBack(dl.out_ref, out_ref);                                                 \
            break;                                                                                 \
        }                                                                                          \
        }                                                                                          \
    }

TEST_PERACT_3D(GPU_Bn3dPerAct_FP32, float)
TEST_PERACT_3D(GPU_Bn3dPerAct_FP16, half_float::half)
TEST_PERACT_3D(GPU_Bn3dPerAct_BFP16, bfloat16)

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Bn3dPerAct_FP32,
                         testing::ValuesIn(GetBN3DPerActTestCases(BN3DPerActTestSet::Full)));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Bn3dPerAct_FP16,
                         testing::ValuesIn(GetBN3DPerActTestCases(BN3DPerActTestSet::Standard)));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Bn3dPerAct_BFP16,
                         testing::ValuesIn(GetBN3DPerActTestCases(BN3DPerActTestSet::Standard)));

} // namespace
