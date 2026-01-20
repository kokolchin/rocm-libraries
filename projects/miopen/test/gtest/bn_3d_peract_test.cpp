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

    std::vector<std::size_t> GetInput() const { return {n, c, d, h, w}; }
};

std::vector<BN3DPerActTestCase> GetBN3DPerActTestCases()
{
    std::vector<BN3DPerActTestCase> test_cases;
    std::vector<BN3DPerActTestType> types = {BN3DPerActTestType::ForwardTraining,
                                             BN3DPerActTestType::ForwardInferenceRecalc,
                                             BN3DPerActTestType::ForwardInferenceUseEstimated,
                                             BN3DPerActTestType::BackwardRecalc,
                                             BN3DPerActTestType::BackwardUseSaved};

    for(const auto& shape : get_3d_bn_peract_inputs(4))
    {
        for(const auto& type : types)
        {
            test_cases.push_back({shape[0], shape[1], shape[2], shape[3], shape[4], type});
        }
    }
    return test_cases;
}
} // namespace

template <typename T>
struct GPU_Bn3dPerAct : public ::testing::TestWithParam<BN3DPerActTestCase>
{
    using AccDataType = std::conditional_t<std::is_same_v<T, double>, double, float>;

    void SetUp() override
    {
        const auto& tc = this->GetParam();
        n              = tc.n;
        c              = tc.c;
        d              = tc.d;
        h              = tc.h;
        w              = tc.w;

        if(n == 1 && (tc.test_type == BN3DPerActTestType::ForwardTraining ||
                      tc.test_type == BN3DPerActTestType::BackwardRecalc ||
                      tc.test_type == BN3DPerActTestType::BackwardUseSaved))
        {
            GTEST_SKIP() << "Batch size of 1 is not supported for BN training/backward";
        }

        auto&& handle = get_handle();

        input   = tensor<T>{n, c, d, h, w};
        output  = tensor<T>{n, c, d, h, w};
        out_ref = tensor<AccDataType>{n, c, d, h, w};

        input.generate(uniform_signed_initializer<T>(2e-3, 1000));

        miopen::DeriveBNTensorDescriptor(derivedBnDesc, input.desc, miopenBNPerActivation);

        scale   = tensor<AccDataType>{derivedBnDesc.GetLengths()};
        shift   = tensor<AccDataType>{derivedBnDesc.GetLengths()};
        runMean = tensor<AccDataType>{derivedBnDesc.GetLengths()};
        runVar  = tensor<AccDataType>{derivedBnDesc.GetLengths()};

        scale.generate(uniform_signed_initializer<AccDataType>(2e-3, 1000));
        shift.generate(uniform_signed_initializer<AccDataType>(2e-3, 1000));
        runMean.generate(uniform_signed_initializer<AccDataType>(2e-3, 1000));
        runVar.generate(uniform_unsigned_initializer<AccDataType>(2e-3, 1000));

        in_dev      = handle.Write(input.data);
        scale_dev   = handle.Write(scale.data);
        shift_dev   = handle.Write(shift.data);
        runMean_dev = handle.Write(runMean.data);
        runVar_dev  = handle.Write(runVar.data);
        out_dev     = handle.Write(output.data);

        if(std::is_same_v<T, float> || std::is_same_v<T, double>)
            tolerance = 1e-5;
        else if(std::is_same_v<T, bfloat16>)
            tolerance = 1e-2;
        else
            tolerance = 5e-3;
    }

    // Helper for ComputeCPUBN* functions from test_operations.hpp
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

        miopenBatchNormMode_t bn_mode = miopenBNPerActivation;
        double epsilon                = MIO_BN_TEST_EPSILON;
        double averageFactor          = MIO_BN_TEST_EXPAVGFACTOR;
        bool useInverseVariance       = false;
    };

    std::size_t n, c, d, h, w;
    tensor<T> input;
    tensor<T> output;
    tensor<AccDataType> out_ref;
    tensor<AccDataType> scale;
    tensor<AccDataType> shift;
    tensor<AccDataType> runMean;
    tensor<AccDataType> runVar;
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
    double tolerance    = 5e-3;
};
using GPU_Bn3dPerAct_FP32 = GPU_Bn3dPerAct<float>;
using GPU_Bn3dPerAct_FP16 = GPU_Bn3dPerAct<half_float::half>;
using GPU_Bn3dPerAct_BF16 = GPU_Bn3dPerAct<bfloat16>;
using GPU_Bn3dPerAct_FP64 = GPU_Bn3dPerAct<double>;
using GPU_Bn3dPerAct_INT8 = GPU_Bn3dPerAct<int8_t>;

#define TEST_PERACT_3D(fixture, data_type)                                                         \
    TEST_P(fixture, Test)                                                                          \
    {                                                                                              \
        const auto& test_case = this->GetParam();                                                  \
        auto&& handle         = get_handle();                                                      \
                                                                                                   \
        if(std::is_same_v<data_type, int8_t> &&                                                    \
           (test_case.test_type == BN3DPerActTestType::ForwardTraining ||                          \
            test_case.test_type == BN3DPerActTestType::BackwardRecalc ||                           \
            test_case.test_type == BN3DPerActTestType::BackwardUseSaved))                          \
        {                                                                                          \
            GTEST_SKIP() << "INT8 is only supported for inference";                                 \
        }                                                                                          \
                                                                                                   \
        switch(test_case.test_type)                                                                \
        {                                                                                          \
        case BN3DPerActTestType::ForwardTraining: {                                                \
            tensor<AccDataType> saveMean{derivedBnDesc.GetLengths()};                              \
            tensor<AccDataType> saveInvVar{derivedBnDesc.GetLengths()};                            \
            auto saveMean_dev   = handle.Write(saveMean.data);                                     \
            auto saveInvVar_dev = handle.Write(saveInvVar.data);                                   \
                                                                                                   \
            miopenStatus_t status = miopenBatchNormalizationForwardTraining(                       \
                &handle,                                                                           \
                miopenBNPerActivation,                                                             \
                &alpha,                                                                            \
                &beta,                                                                             \
                &input.desc,                                                                       \
                in_dev.get(),                                                                      \
                &output.desc,                                                                      \
                out_dev.get(),                                                                     \
                &derivedBnDesc,                                                                    \
                scale_dev.get(),                                                                   \
                shift_dev.get(),                                                                   \
                expAvgFactor,                                                                      \
                runMean_dev.get(),                                                                 \
                runVar_dev.get(),                                                                  \
                epsilon,                                                                           \
                saveMean_dev.get(),                                                                \
                saveInvVar_dev.get());                                                             \
                                                                                                   \
            ASSERT_EQ(status, miopenStatusSuccess);                                                \
                                                                                                   \
            output.data     = handle.Read<data_type>(out_dev, output.data.size());                 \
            saveMean.data   = handle.Read<AccDataType>(saveMean_dev, saveMean.data.size());        \
            saveInvVar.data = handle.Read<AccDataType>(saveInvVar_dev, saveInvVar.data.size());    \
            runMean.data    = handle.Read<AccDataType>(runMean_dev, runMean.data.size());          \
            runVar.data     = handle.Read<AccDataType>(runVar_dev, runVar.data.size());            \
                                                                                                   \
            typename GPU_Bn3dPerAct<data_type>::DLModule dl;                                       \
            dl.input            = input;                                                           \
            dl.output           = output;                                                          \
            dl.out_ref          = out_ref;                                                         \
            dl.scale            = scale;                                                           \
            dl.shift            = shift;                                                           \
            dl.saveMean_ref     = saveMean;                                                        \
            dl.saveVariance_ref = saveInvVar;                                                      \
            dl.runMean_ref      = runMean;                                                         \
            dl.runVariance_ref  = runVar;                                                          \
                                                                                                   \
            test::ComputeCPUBNFwdTrain(dl);                                                        \
            test::CompareTensor(output, dl.out_ref, tolerance);                                    \
            test::CompareTensor(saveMean, dl.saveMean_ref, tolerance);                             \
            test::CompareTensor(saveInvVar, dl.saveVariance_ref, tolerance);                       \
            test::CompareTensor(runMean, dl.runMean_ref, tolerance);                               \
            test::CompareTensor(runVar, dl.runVariance_ref, tolerance);                            \
            break;                                                                                 \
        }                                                                                          \
        case BN3DPerActTestType::ForwardInferenceRecalc:                                           \
        case BN3DPerActTestType::ForwardInferenceUseEstimated: {                                   \
            miopenStatus_t status = miopenBatchNormalizationForwardInference(&handle,              \
                                                                             miopenBNPerActivation, \
                                                                             &alpha,               \
                                                                             &beta,                \
                                                                             &input.desc,          \
                                                                             in_dev.get(),         \
                                                                             &output.desc,         \
                                                                             out_dev.get(),        \
                                                                             &derivedBnDesc,       \
                                                                             scale_dev.get(),      \
                                                                             shift_dev.get(),      \
                                                                             runMean_dev.get(),    \
                                                                             runVar_dev.get(),     \
                                                                             epsilon);             \
                                                                                                   \
            ASSERT_EQ(status, miopenStatusSuccess);                                                \
                                                                                                   \
            output.data = handle.Read<data_type>(out_dev, output.data.size());                     \
                                                                                                   \
            typename GPU_Bn3dPerAct<data_type>::DLModule dl;                                       \
            dl.input              = input;                                                         \
            dl.output             = output;                                                        \
            dl.out_ref            = out_ref;                                                       \
            dl.scale              = scale;                                                         \
            dl.shift              = shift;                                                         \
            dl.estMean            = runMean;                                                       \
            dl.estVariance        = runVar;                                                        \
            dl.useInverseVariance = false;                                                         \
                                                                                                   \
            test::ComputeCPUBNInference(dl);                                                       \
            test::CompareTensor(output, dl.out_ref, tolerance);                                    \
            break;                                                                                 \
        }                                                                                          \
        case BN3DPerActTestType::BackwardRecalc: {                                                 \
            tensor<data_type> dy_input{n, c, d, h, w};                                             \
            dy_input.generate(uniform_signed_initializer<data_type>(2e-3, 1000));                  \
            auto dy_dev = handle.Write(dy_input.data);                                             \
                                                                                                   \
            tensor<data_type> dx_output{n, c, d, h, w};                                            \
            tensor<AccDataType> dscale{derivedBnDesc.GetLengths()};                                \
            tensor<AccDataType> dshift{derivedBnDesc.GetLengths()};                                \
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
            dx_output.data = handle.Read<data_type>(dx_dev, dx_output.data.size());                \
            dscale.data    = handle.Read<AccDataType>(dscale_dev, dscale.data.size());             \
            dshift.data    = handle.Read<AccDataType>(dshift_dev, dshift.data.size());             \
                                                                                                   \
            typename GPU_Bn3dPerAct<data_type>::DLModule dl;                                       \
            dl.input      = input;                                                                 \
            dl.output     = output;                                                                \
            dl.dy         = dy_input;                                                              \
            dl.out_ref    = out_ref;                                                               \
            dl.bnScale    = scale;                                                                 \
            dl.dScale_ref = dscale;                                                                \
            dl.dBias_ref  = dshift;                                                                \
                                                                                                   \
            typename GPU_Bn3dPerAct<data_type>::DLModule dl_fwd;                                   \
            dl_fwd.input            = input;                                                       \
            dl_fwd.output           = output;                                                      \
            dl_fwd.out_ref          = out_ref;                                                     \
            dl_fwd.scale            = scale;                                                       \
            dl_fwd.shift            = shift;                                                       \
            dl_fwd.saveMean_ref     = tensor<AccDataType>{derivedBnDesc.GetLengths()};             \
            dl_fwd.saveVariance_ref = tensor<AccDataType>{derivedBnDesc.GetLengths()};             \
            dl_fwd.runMean_ref      = runMean;                                                     \
            dl_fwd.runVariance_ref  = runVar;                                                      \
            test::ComputeCPUBNFwdTrain(dl_fwd);                                                    \
            dl.savedMean   = dl_fwd.saveMean_ref;                                                  \
            dl.savedInvVar = dl_fwd.saveVariance_ref;                                              \
                                                                                                   \
            test::ComputeCPUBNBwd(dl);                                                             \
            test::CompareTensor(dx_output, dl.out_ref, tolerance);                                 \
            test::CompareTensor(dscale, dl.dScale_ref, tolerance);                                 \
            test::CompareTensor(dshift, dl.dBias_ref, tolerance);                                  \
            break;                                                                                 \
        }                                                                                          \
        case BN3DPerActTestType::BackwardUseSaved: {                                               \
            tensor<AccDataType> saveMean{derivedBnDesc.GetLengths()};                              \
            tensor<AccDataType> saveInvVar{derivedBnDesc.GetLengths()};                            \
            auto saveMean_dev   = handle.Write(saveMean.data);                                     \
            auto saveInvVar_dev = handle.Write(saveInvVar.data);                                   \
                                                                                                   \
            miopenStatus_t status = miopenBatchNormalizationForwardTraining(                       \
                &handle,                                                                           \
                miopenBNPerActivation,                                                             \
                &alpha,                                                                            \
                &beta,                                                                             \
                &input.desc,                                                                       \
                in_dev.get(),                                                                      \
                &output.desc,                                                                      \
                out_dev.get(),                                                                     \
                &derivedBnDesc,                                                                    \
                scale_dev.get(),                                                                   \
                shift_dev.get(),                                                                   \
                expAvgFactor,                                                                      \
                runMean_dev.get(),                                                                 \
                runVar_dev.get(),                                                                  \
                epsilon,                                                                           \
                saveMean_dev.get(),                                                                \
                saveInvVar_dev.get());                                                             \
                                                                                                   \
            ASSERT_EQ(status, miopenStatusSuccess);                                                \
                                                                                                   \
            saveMean.data   = handle.Read<AccDataType>(saveMean_dev, saveMean.data.size());        \
            saveInvVar.data = handle.Read<AccDataType>(saveInvVar_dev, saveInvVar.data.size());    \
                                                                                                   \
            tensor<data_type> dy_input{n, c, d, h, w};                                             \
            dy_input.generate(uniform_signed_initializer<data_type>(2e-3, 1000));                  \
            auto dy_dev = handle.Write(dy_input.data);                                             \
                                                                                                   \
            tensor<data_type> dx_output{n, c, d, h, w};                                            \
            tensor<AccDataType> dscale{derivedBnDesc.GetLengths()};                                \
            tensor<AccDataType> dshift{derivedBnDesc.GetLengths()};                                \
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
            dx_output.data = handle.Read<data_type>(dx_dev, dx_output.data.size());                \
            dscale.data    = handle.Read<AccDataType>(dscale_dev, dscale.data.size());             \
            dshift.data    = handle.Read<AccDataType>(dshift_dev, dshift.data.size());             \
                                                                                                   \
            typename GPU_Bn3dPerAct<data_type>::DLModule dl;                                       \
            dl.input       = input;                                                                \
            dl.output      = output;                                                               \
            dl.dy          = dy_input;                                                             \
            dl.out_ref     = out_ref;                                                              \
            dl.bnScale     = scale;                                                                \
            dl.dScale_ref  = dscale;                                                               \
            dl.dBias_ref   = dshift;                                                               \
            dl.savedMean   = saveMean;                                                             \
            dl.savedInvVar = saveInvVar;                                                           \
                                                                                                   \
            test::ComputeCPUBNBwd(dl);                                                             \
            test::CompareTensor(dx_output, dl.out_ref, tolerance);                                 \
            test::CompareTensor(dscale, dl.dScale_ref, tolerance);                                 \
            test::CompareTensor(dshift, dl.dBias_ref, tolerance);                                  \
            break;                                                                                 \
        }                                                                                          \
        }                                                                                          \
    }

TEST_PERACT_3D(GPU_Bn3dPerAct_FP32, float)
TEST_PERACT_3D(GPU_Bn3dPerAct_FP16, half_float::half)
TEST_PERACT_3D(GPU_Bn3dPerAct_BF16, bfloat16)
TEST_PERACT_3D(GPU_Bn3dPerAct_FP64, double)
TEST_PERACT_3D(GPU_Bn3dPerAct_INT8, int8_t)

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Bn3dPerAct_FP32, testing::ValuesIn(GetBN3DPerActTestCases()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Bn3dPerAct_FP16, testing::ValuesIn(GetBN3DPerActTestCases()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Bn3dPerAct_BF16, testing::ValuesIn(GetBN3DPerActTestCases()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Bn3dPerAct_FP64, testing::ValuesIn(GetBN3DPerActTestCases()));
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Bn3dPerAct_INT8, testing::ValuesIn(GetBN3DPerActTestCases()));