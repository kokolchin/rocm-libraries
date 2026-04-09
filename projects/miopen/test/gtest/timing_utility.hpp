#ifndef GUARD_MIOPEN_TEST_GTEST_TIMING_UTILITY_HPP
#define GUARD_MIOPEN_TEST_GTEST_TIMING_UTILITY_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
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

struct TimerCounter
{
    std::atomic<std::int64_t> total_us{0};
    std::atomic<std::uint64_t> calls{0};
};

struct TimerRow
{
    const char* name;
    std::uint64_t calls;
    double total_ms;
};

constexpr std::array<const char*, 17> kHostNames = {
    "run_pooling_test_with_index_type.total",
    "run_pooling_test_with_index_type.forward",
    "run_pooling_test_with_index_type.backward",
    "verify_forward_pooling.cpu.total",
    "verify_forward_pooling.gpu.total",
    "verify_backward_pooling.cpu.total",
    "verify_backward_pooling.gpu.total",
    "verify_backward_pooling.cpu.inner_loops.main",
    "verify_backward_pooling.cpu.inner_loops.max_path",
    "verify_backward_pooling.cpu.inner_loops.avg_path",
    "verify_backward_pooling.cpu.index_verify",
    "verify_backward_pooling.gpu.io.write_input",
    "verify_backward_pooling.gpu.io.write_dout",
    "verify_backward_pooling.gpu.io.write_out",
    "verify_backward_pooling.gpu.io.create_din",
    "verify_backward_pooling.gpu.io.workspace_write",
    "verify_backward_pooling.gpu.io.read_dinput",
};

constexpr std::array<const char*, 2> kGpuNames = {
    "verify_forward_pooling.gpu.forward_kernel",
    "verify_backward_pooling.gpu.backward_kernel",
};

inline std::array<TimerCounter, kHostNames.size()>& HostCounters()
{
    static std::array<TimerCounter, kHostNames.size()> counters{};
    return counters;
}

inline std::array<TimerCounter, kGpuNames.size()>& GpuCounters()
{
    static std::array<TimerCounter, kGpuNames.size()> counters{};
    return counters;
}

inline int FindHostIndex(const std::string& name)
{
    for(std::size_t i = 0; i < kHostNames.size(); ++i)
    {
        if(name == kHostNames[i])
            return static_cast<int>(i);
    }
    return -1;
}

inline int FindGpuIndex(const std::string& name)
{
    for(std::size_t i = 0; i < kGpuNames.size(); ++i)
    {
        if(name == kGpuNames[i])
            return static_cast<int>(i);
    }
    return -1;
}

inline std::vector<TimerRow> CollectHostRows()
{
    std::vector<TimerRow> rows;
    rows.reserve(kHostNames.size());
    const auto& counters = HostCounters();
    for(std::size_t i = 0; i < kHostNames.size(); ++i)
    {
        const auto calls = counters[i].calls.load(std::memory_order_relaxed);
        const auto us    = counters[i].total_us.load(std::memory_order_relaxed);
        if(calls == 0 || us == 0)
            continue;
        rows.push_back({kHostNames[i], calls, static_cast<double>(us) / 1000.0});
    }
    return rows;
}

inline std::vector<TimerRow> CollectGpuRows()
{
    std::vector<TimerRow> rows;
    rows.reserve(kGpuNames.size());
    const auto& counters = GpuCounters();
    for(std::size_t i = 0; i < kGpuNames.size(); ++i)
    {
        const auto calls = counters[i].calls.load(std::memory_order_relaxed);
        const auto us    = counters[i].total_us.load(std::memory_order_relaxed);
        if(calls == 0 || us == 0)
            continue;
        rows.push_back({kGpuNames[i], calls, static_cast<double>(us) / 1000.0});
    }
    return rows;
}

inline void WriteTopReport(const std::string& path,
                           const std::string& title,
                           std::vector<TimerRow> rows,
                           std::size_t top_n = 30)
{
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.total_ms > b.total_ms;
    });

    std::ofstream out(path);
    if(!out.is_open())
        return;

    out << title << '\n';
    out << "Name\tCalls\tTotal(ms)\tAvg(ms)\n";
    out << "-----------------------------------------------\n";
    std::size_t emitted = 0;
    for(const auto& row : rows)
    {
        if(emitted >= top_n)
            break;
        const double avg = row.calls == 0 ? 0.0 : row.total_ms / static_cast<double>(row.calls);
        out << row.name << '\t' << row.calls << '\t' << std::fixed << std::setprecision(3)
            << row.total_ms << '\t' << avg << '\n';
        emitted++;
    }
}

inline void ReportAll()
{
    const auto host_rows = CollectHostRows();
    const auto gpu_rows  = CollectGpuRows();
    WriteTopReport("timing_per_function.txt", "Top host sections", host_rows, 30);
    WriteTopReport("timing_gpu_sections.txt", "Top gpu sections", gpu_rows, 30);
    std::ofstream summary("timing_summary.txt");
    if(summary.is_open())
    {
        summary << "Host sections tracked: " << host_rows.size() << '\n';
        summary << "Gpu sections tracked: " << gpu_rows.size() << '\n';
        summary << "See timing_per_function.txt and timing_gpu_sections.txt for top entries.\n";
    }
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
    const int idx = FindHostIndex(name);
    if(idx < 0)
        return;

    auto& c = HostCounters()[static_cast<std::size_t>(idx)];
    c.total_us.fetch_add(static_cast<std::int64_t>(std::llround(ms * 1000.0)),
                         std::memory_order_relaxed);
    c.calls.fetch_add(1, std::memory_order_relaxed);
}

inline void AddGpuSample(const std::string& name, double ms)
{
    RegisterReporterOnce();
    const int idx = FindGpuIndex(name);
    if(idx < 0)
        return;

    auto& c = GpuCounters()[static_cast<std::size_t>(idx)];
    c.total_us.fetch_add(static_cast<std::int64_t>(std::llround(ms * 1000.0)),
                         std::memory_order_relaxed);
    c.calls.fetch_add(1, std::memory_order_relaxed);
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
