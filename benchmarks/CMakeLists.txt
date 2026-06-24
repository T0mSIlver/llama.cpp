# CodSpeed benchmarks. Built only when LLAMA_BUILD_BENCHMARKS=ON.
# Uses the CodSpeed fork of google-benchmark so existing BENCHMARK() macros are
# instrumented for CPU simulation when -DCODSPEED_MODE=simulation is set.

include(FetchContent)

set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL     OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

# Build the benchmark/codspeed stack statically (directory scope only, so it
# does not affect llama/ggml). With BUILD_SHARED_LIBS=ON the codspeed runtime
# symbol g_hooks ends up hidden inside libcodspeed.so and fails to link into the
# benchmark; static linking bakes it in, matching the upstream codspeed example.
set(BUILD_SHARED_LIBS OFF)

FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/CodSpeedHQ/codspeed-cpp
    GIT_TAG        main
    SOURCE_SUBDIR  google_benchmark
)
FetchContent_MakeAvailable(google_benchmark)

add_executable(bench-tokenizer bench-tokenizer.cpp)
target_link_libraries(bench-tokenizer PRIVATE llama benchmark::benchmark)
target_compile_definitions(bench-tokenizer PRIVATE
    VOCAB_PATH="${CMAKE_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf")
