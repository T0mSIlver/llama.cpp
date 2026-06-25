#include <benchmark/benchmark.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Gated DeltaNet (GGML_OP_GATED_DELTA_NET) is the linear-attention kernel used by
// Qwen3-Next / Qwen3.5 / Kimi-Linear. The fused op is the default path for both
// decode and chunked prefill (cparams.fused_gdn_ar / fused_gdn_ch default to true).
//
// Dimensions mirror the Qwen3-Next gated-delta-net config: head_size (S_v) = 128,
// head_count (n_v_heads) = 32. n_seq_tokens models a prefill chunk; the recurrence
// runs sequentially over those tokens per head.
namespace {

constexpr int64_t S_v          = 128; // head_size (== ssm_d_state)
constexpr int64_t H            = 32;  // head_count (n_v_heads)
constexpr int64_t n_seq_tokens = 64;  // prefill chunk length
constexpr int64_t n_seqs       = 1;
constexpr int64_t K            = 1;   // keep final state only

// kda == false: scalar per-head gate g[1, H, T, S]   (Qwen3-Next)
// kda == true : per-channel gate    g[S_v, H, T, S]   (Qwen3.5 KDA / Kimi-Linear)
struct gdn_fixture {
    ggml_backend_t  backend = nullptr;
    ggml_context *  ctx     = nullptr;
    ggml_gallocr_t  allocr  = nullptr;
    ggml_cgraph *   gf      = nullptr;

    explicit gdn_fixture(bool kda) {
        const size_t ctx_size = ggml_tensor_overhead()*32 + ggml_graph_overhead() + 1024;
        ctx = ggml_init({ ctx_size, nullptr, /* no_alloc = */ true });

        const int64_t g_ne0 = kda ? S_v : 1;

        ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v,   H, n_seq_tokens, n_seqs);
        ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v,   H, n_seq_tokens, n_seqs);
        ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v,   H, n_seq_tokens, n_seqs);
        ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g_ne0, H, n_seq_tokens, n_seqs);
        ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,     H, n_seq_tokens, n_seqs);
        ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v, H, n_seqs);

        for (ggml_tensor * t : { q, k, v, g, beta, state }) {
            ggml_set_input(t);
        }

        ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, K);
        ggml_set_output(out);

        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        backend = ggml_backend_cpu_init();
        // single thread: deterministic instruction counts for CodSpeed simulation
        ggml_backend_cpu_set_n_threads(backend, 1);

        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(allocr, gf)) {
            fprintf(stderr, "bench-gated-delta-net: failed to allocate graph\n");
            std::abort();
        }

        std::mt19937 rng(1234);
        fill(q,     rng, -1.0f,   1.0f);
        fill(k,     rng, -1.0f,   1.0f);
        fill(v,     rng, -0.3f,   5.0f);
        fill(g,     rng, -20.0f, -1e-4f); // negative => exp(g) in (0,1], no overflow
        fill(beta,  rng,  0.0f,   1.0f);
        fill(state, rng, -1.0f,   1.0f);
    }

    static void fill(ggml_tensor * t, std::mt19937 & rng, float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        std::vector<float> buf(ggml_nelements(t));
        for (float & x : buf) {
            x = dist(rng);
        }
        ggml_backend_tensor_set(t, buf.data(), 0, ggml_nbytes(t));
    }

    ~gdn_fixture() {
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
};

} // namespace

static void run(benchmark::State & state, bool kda) {
    gdn_fixture f(kda);

    // warm up + sanity check that the graph computes without error
    if (ggml_backend_graph_compute(f.backend, f.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "bench-gated-delta-net: compute failed\n");
        std::abort();
    }

    for (auto _ : state) {
        ggml_backend_graph_compute(f.backend, f.gf);
        benchmark::ClobberMemory();
    }

    state.SetLabel(std::to_string(H) + " heads x " + std::to_string(n_seq_tokens) +
                   " tokens, S_v=" + std::to_string(S_v));
}

static void BM_gated_delta_net_scalar_gate(benchmark::State & state) { run(state, /* kda = */ false); }
static void BM_gated_delta_net_kda        (benchmark::State & state) { run(state, /* kda = */ true ); }

BENCHMARK(BM_gated_delta_net_scalar_gate);
BENCHMARK(BM_gated_delta_net_kda);

BENCHMARK_MAIN();
