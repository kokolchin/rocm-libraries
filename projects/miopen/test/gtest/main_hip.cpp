// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <chrono>
#include <iostream>
#include "function_timer.hpp"

// This test event listener ensures that HIP errors are cleaned up after every test, and will flag
// tests that don't clean up their own errors
class HIPErrorHandler : public testing::EmptyTestEventListener
{
public:
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        FunctionTimer ft("HIPErrorHandler::OnTestEnd");
        auto hipError    = hipGetLastError();
        auto hipExtError = hipExtGetLastError();

        EXPECT_EQ(hipError, hipSuccess)
            << " hipGetLastError returned error code " << hipError << " after test "
            << test_info.test_suite_name() << "." << test_info.name()
            << ". Error string: " << hipGetErrorString(hipError);
        EXPECT_EQ(hipExtError, hipSuccess)
            << " hipExtGetLastError returned error code " << hipExtError << " after test "
            << test_info.test_suite_name() << "." << test_info.name()
            << ". Error string: " << hipGetErrorString(hipExtError);
    }
};

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new HIPErrorHandler);

    int result = RUN_ALL_TESTS();
    
    FunctionTimer::report();

    return result;
}
