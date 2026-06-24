# CodSpeed performance benchmarks.
#
# Uses CodSpeed's fork of Google Benchmark so the same benchmarks can run
# both locally (standard Google Benchmark output) and under CodSpeed's
# instruments in CI (controlled by -DCODSPEED_MODE=...).

include(FetchContent)

# Build Google Benchmark and the CodSpeed core as static libraries, regardless
# of the project-wide BUILD_SHARED_LIBS setting (ON by default here because of
# ggml). As shared libraries the CodSpeed instrumentation symbols are not
# exported, which breaks linking of the benchmark executable.
set(BUILD_SHARED_LIBS OFF)

FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/CodSpeedHQ/codspeed-cpp
    SOURCE_SUBDIR   google_benchmark
    GIT_TAG         v2.4.0
)

set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(google_benchmark)

add_executable(llama-codspeed-bench bench-quantize.cpp)
target_link_libraries(llama-codspeed-bench PRIVATE ggml benchmark::benchmark)
