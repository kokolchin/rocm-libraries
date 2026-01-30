// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <chrono>
#include <iostream>

// This test event listener ensures that HIP errors are cleaned up after every test, and will flag
// tests that don't clean up their own errors
class HIPErrorHandler : public testing::EmptyTestEventListener
{
public:
    static std::chrono::nanoseconds total_hip_error_check_time;

    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto hipError    = hipGetLastError();
        auto hipExtError = hipExtGetLastError();
        auto end = std::chrono::high_resolution_clock::now();
        
        total_hip_error_check_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

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

std::chrono::nanoseconds HIPErrorHandler::total_hip_error_check_time{0};

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new HIPErrorHandler);

    int result = RUN_ALL_TESTS();
    
    std::cout << "Total time spent in hipGetLastError/hipExtGetLastError: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(HIPErrorHandler::total_hip_error_check_time).count() 
              << " ms" << std::endl;

    return result;
}
