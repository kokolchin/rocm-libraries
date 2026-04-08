#ifndef GUARD_MIOPEN_TEST_GTEST_TIMING_UTILITY_HPP
#define GUARD_MIOPEN_TEST_GTEST_TIMING_UTILITY_HPP

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#if __has_include(<hip/hip_runtime_api.h>)
#include <hip/hip_runtime_api.h>
#define MIOPEN_POOLING_TIMING_HAS_HIP 1
#else
#define MIOPEN_POOLING_TIMING_HAS_HIP 0
#endif

namespace pooling_gtest {
namespace timing_utility {

#if MIOPEN_POOLING_TIMING_HAS_HIP
inline void ConsumeHipStatus([[maybe_unused]] hipError_t status) {}
#endif

struct TimerStat
{
    double total_ms = 0.0;
    std::size_t calls = 0;
};

inline std::map<std::string, TimerStat>& HostStats()
{
    static std::map<std::string, TimerStat> stats;
    return stats;
}

inline std::map<std::string, TimerStat>& GpuStats()
{
    static std::map<std::string, TimerStat> stats;
    return stats;
}

inline std::mutex& StatsMutex()
{
    static std::mutex m;
    return m;
}

inline void WriteReport(const std::string& path, const std::map<std::string, TimerStat>& stats)
{
    std::vector<std::pair<std::string, TimerStat>> rows(stats.begin(), stats.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.second.total_ms > b.second.total_ms;
    });

    std::ofstream out(path);
    if(!out.is_open())
        return;

    out << "Name\tCalls\tTotal(ms)\tAvg(ms)\n";
    out << "-----------------------------------------------\n";
    for(const auto& row : rows)
    {
        const auto& name = row.first;
        const auto& s    = row.second;
        const double avg = s.calls == 0 ? 0.0 : s.total_ms / static_cast<double>(s.calls);
        out << name << '\t' << s.calls << '\t' << std::fixed << std::setprecision(3) << s.total_ms
            << '\t' << avg << '\n';
    }
}

inline void ReportAll()
{
    std::lock_guard<std::mutex> lock(StatsMutex());
    WriteReport("timing_per_function.txt", HostStats());
    WriteReport("timing_gpu_sections.txt", GpuStats());
}

inline void RegisterReporterOnce()
{
    static const bool registered = [] {
        std::atexit(ReportAll);
        return true;
    }();
    (void)registered;
}

inline void AddHostSample(const std::string& name, double ms)
{
    RegisterReporterOnce();
    std::lock_guard<std::mutex> lock(StatsMutex());
    auto& s = HostStats()[name];
    s.total_ms += ms;
    s.calls++;
}

inline void AddGpuSample(const std::string& name, double ms)
{
    RegisterReporterOnce();
    std::lock_guard<std::mutex> lock(StatsMutex());
    auto& s = GpuStats()[name];
    s.total_ms += ms;
    s.calls++;
}

class ScopedTimer
{
public:
    explicit ScopedTimer(std::string name)
        : mName(std::move(name)), mStart(std::chrono::steady_clock::now())
    {
    }

    ~ScopedTimer()
    {
        const auto end = std::chrono::steady_clock::now();
        const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(end - mStart).count();
        AddHostSample(mName, static_cast<double>(us) / 1000.0);
    }

private:
    std::string mName;
    std::chrono::steady_clock::time_point mStart;
};

class ScopedGpuTimer
{
public:
    explicit ScopedGpuTimer(std::string name) : mName(std::move(name))
    {
#if MIOPEN_POOLING_TIMING_HAS_HIP
        ConsumeHipStatus(hipEventCreate(&mStartEvent));
        ConsumeHipStatus(hipEventCreate(&mStopEvent));
        ConsumeHipStatus(hipEventRecord(mStartEvent, nullptr));
#else
        mStart = std::chrono::steady_clock::now();
#endif
    }

    ~ScopedGpuTimer()
    {
#if MIOPEN_POOLING_TIMING_HAS_HIP
        ConsumeHipStatus(hipEventRecord(mStopEvent, nullptr));
        ConsumeHipStatus(hipEventSynchronize(mStopEvent));
        float ms = 0.0f;
        ConsumeHipStatus(hipEventElapsedTime(&ms, mStartEvent, mStopEvent));
        ConsumeHipStatus(hipEventDestroy(mStartEvent));
        ConsumeHipStatus(hipEventDestroy(mStopEvent));
        AddGpuSample(mName, static_cast<double>(ms));
#else
        const auto end = std::chrono::steady_clock::now();
        const auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(end - mStart).count();
        AddGpuSample(mName, static_cast<double>(us) / 1000.0);
#endif
    }

private:
    std::string mName;
#if MIOPEN_POOLING_TIMING_HAS_HIP
    hipEvent_t mStartEvent{};
    hipEvent_t mStopEvent{};
#else
    std::chrono::steady_clock::time_point mStart;
#endif
};

} // namespace timing_utility
} // namespace pooling_gtest

#if defined(MIOPEN_POOLING_TIMING)
#define MIOPEN_POOLING_CONCAT_IMPL(a, b) a##b
#define MIOPEN_POOLING_CONCAT(a, b) MIOPEN_POOLING_CONCAT_IMPL(a, b)
#define POOLING_TIMED_SCOPE(name)                                                             \
    ::pooling_gtest::timing_utility::ScopedTimer MIOPEN_POOLING_CONCAT(_pooling_timer_,      \
                                                                        __LINE__)(name)
#define POOLING_GPU_TIMED_SCOPE(name)                                                         \
    ::pooling_gtest::timing_utility::ScopedGpuTimer MIOPEN_POOLING_CONCAT(_pooling_gpu_timer_, \
                                                                           __LINE__)(name)
#else
#define POOLING_TIMED_SCOPE(name)
#define POOLING_GPU_TIMED_SCOPE(name)
#endif

#endif
