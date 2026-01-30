/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_TEST_GTEST_FUNCTION_TIMER_HPP
#define GUARD_MIOPEN_TEST_GTEST_FUNCTION_TIMER_HPP

#include <map>
#include <string>
#include <ctime>
#include <iostream>
#include <mutex>

struct FunctionTimer 
{
    FunctionTimer(char const * name) : mName(name), mStartTime(clock()) { }
    ~FunctionTimer() 
    { 
        std::lock_guard<std::mutex> lock(get_mutex());
        get_map()[mName] += clock() - mStartTime; 
    }

    static void report()
    {
        std::lock_guard<std::mutex> lock(get_mutex());
        for (auto const& [name, ticks] : get_map())
        {
            std::cout << "[PROFILING] " << name << " --> " << ( (float)(ticks) / CLOCKS_PER_SEC ) << " sec" << std::endl;
        }
    }

    static std::map<std::string, clock_t>& get_map()
    {
        static std::map<std::string, clock_t> mFunctionTimes;
        return mFunctionTimes;
    }

    static std::mutex& get_mutex()
    {
        static std::mutex m;
        return m;
    }

    std::string mName;
    clock_t mStartTime;
};

#endif
