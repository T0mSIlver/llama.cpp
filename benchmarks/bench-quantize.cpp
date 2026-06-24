// CodSpeed benchmarks for ggml quantization kernels.
//
// These benchmarks exercise CPU-bound hot paths that are central to
// llama.cpp inference: quantizing activations from float, dequantizing
// weights back to float, and the quantized vector dot product. They run
// on synthetic data and require no model files, which keeps them fast and
// fully deterministic under CodSpeed's CPU simulation instrument.

#include "ggml.h"
#include "ggml-cpu.h"

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
#include <vector>

// Number of float values processed per benchmark iteration.
// 4096 is a multiple of 256, so it is a valid row length for every
// block-quantized type used here (block sizes 32 and 256).
static constexpr int64_t N = 4096;

static std::vector<float> make_data(int64_t n, float offset) {
    std::vector<float> data(n);
    for (int64_t i = 0; i < n; i++) {
        data[i] = 0.1f + 2.0f * std::cos((float) i + offset);
    }
    return data;
}

// Quantize a row of floats into the given ggml type (from_float).
static void bench_quantize(benchmark::State & state, ggml_type type) {
    ggml_cpu_init();
    ggml_quantize_init(type);

    const auto * traits_cpu = ggml_get_type_traits_cpu(type);

    const std::vector<float> src = make_data(N, 0.0f);
    std::vector<char>        dst(ggml_row_size(type, N));

    for (auto _ : state) {
        traits_cpu->from_float(src.data(), dst.data(), N);
        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }
}

// Dequantize a quantized row back to floats (to_float).
static void bench_dequantize(benchmark::State & state, ggml_type type) {
    ggml_cpu_init();
    ggml_quantize_init(type);

    const auto * traits     = ggml_get_type_traits(type);
    const auto * traits_cpu = ggml_get_type_traits_cpu(type);

    const std::vector<float> src = make_data(N, 0.0f);
    std::vector<char>        q(ggml_row_size(type, N));
    std::vector<float>       dst(N);

    traits_cpu->from_float(src.data(), q.data(), N);

    for (auto _ : state) {
        traits->to_float(q.data(), dst.data(), N);
        benchmark::DoNotOptimize(dst.data());
        benchmark::ClobberMemory();
    }
}

// Quantized vector dot product (vec_dot), the core of matrix multiplication.
static void bench_vec_dot(benchmark::State & state, ggml_type type) {
    ggml_cpu_init();
    ggml_quantize_init(type);

    const auto * traits_cpu = ggml_get_type_traits_cpu(type);
    const auto * vdot       = ggml_get_type_traits_cpu(traits_cpu->vec_dot_type);

    const std::vector<float> src1 = make_data(N, 0.0f);
    const std::vector<float> src2 = make_data(N, 1.0f);

    std::vector<char> q1(ggml_row_size(type, N));
    std::vector<char> q2(ggml_row_size(traits_cpu->vec_dot_type, N));

    traits_cpu->from_float(src1.data(), q1.data(), N);
    vdot->from_float(src2.data(), q2.data(), N);

    for (auto _ : state) {
        float result = 0.0f;
        traits_cpu->vec_dot((int) N, &result, 0, q1.data(), 0, q2.data(), 0, 1);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK_CAPTURE(bench_quantize,   q4_0, GGML_TYPE_Q4_0);
BENCHMARK_CAPTURE(bench_quantize,   q8_0, GGML_TYPE_Q8_0);
BENCHMARK_CAPTURE(bench_quantize,   q4_K, GGML_TYPE_Q4_K);

BENCHMARK_CAPTURE(bench_dequantize, q4_0, GGML_TYPE_Q4_0);
BENCHMARK_CAPTURE(bench_dequantize, q8_0, GGML_TYPE_Q8_0);
BENCHMARK_CAPTURE(bench_dequantize, q4_K, GGML_TYPE_Q4_K);

BENCHMARK_CAPTURE(bench_vec_dot,    q4_0, GGML_TYPE_Q4_0);
BENCHMARK_CAPTURE(bench_vec_dot,    q8_0, GGML_TYPE_Q8_0);
BENCHMARK_CAPTURE(bench_vec_dot,    q4_K, GGML_TYPE_Q4_K);

BENCHMARK_MAIN();
