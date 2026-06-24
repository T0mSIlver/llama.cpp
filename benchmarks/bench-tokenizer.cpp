#include <benchmark/benchmark.h>

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef VOCAB_PATH
#define VOCAB_PATH "models/ggml-vocab-qwen35.gguf"
#endif

// The qwen35 pre-tokenizer (Qwen 3.5 / 3.6, Step-3.5, MiniCPM-V-4.6) differs
// from qwen2 in how it groups Unicode combining marks (\p{M}). The corpus mixes
// Latin prose, source code, digits and scripts that rely on combining marks
// (Vietnamese, Arabic, Devanagari) plus emoji and CJK to exercise that path.
static const char * k_corpus_unit =
    "The quick brown fox jumps over the lazy dog, then writes some code.\n"
    "def tokenize(text):\n    return [t for t in re.split(r'\\s+', text) if t]\n"
    "Numbers and symbols: 1234567890, 3.14159, 0xDEADBEEF, 1e-9, a==b && c!=d.\n"
    "Tieng Viet: Xin chao, toi dang thu nghiem bo ma hoa van ban nay.\n"
    "Tiếng Việt: Xin chào, tôi đang thử nghiệm bộ mã hoá văn bản này.\n"
    "العربية: مرحبا، هذا اختبار بسيط للترميز اللغوي.\n"
    "हिन्दी: नमस्ते, यह एक छोटा परीक्षण है जो संयोजी चिह्नों का उपयोग करता है।\n"
    "Mixed: 🚀 😶‍🌫️ ✅ 我想在 apple 工作 1314151 天 ～ end.\n";

namespace {

struct vocab_fixture {
    llama_model      * model = nullptr;
    const llama_vocab * vocab = nullptr;
    std::string        text;

    vocab_fixture() {
        // silence the model-loader logging so the benchmark output stays clean
        llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
        llama_backend_init();

        auto mparams = llama_model_default_params();
        mparams.vocab_only = true;

        model = llama_model_load_from_file(VOCAB_PATH, mparams);
        if (model == nullptr) {
            fprintf(stderr, "bench-tokenizer: failed to load vocab '%s'\n", VOCAB_PATH);
            std::abort();
        }
        vocab = llama_model_get_vocab(model);

        // repeat the unit to build a stable, non-trivial workload (~tens of KB)
        for (int i = 0; i < 96; ++i) {
            text += k_corpus_unit;
        }
    }

    ~vocab_fixture() {
        llama_model_free(model);
        llama_backend_free();
    }
};

const vocab_fixture & fixture() {
    static vocab_fixture f;
    return f;
}

} // namespace

static void BM_tokenize_qwen35(benchmark::State & state) {
    const vocab_fixture & f = fixture();

    // tokens <= bytes, so this buffer never needs to grow inside the loop
    std::vector<llama_token> tokens(f.text.size() + 16);

    int32_t n = 0;
    for (auto _ : state) {
        n = llama_tokenize(f.vocab, f.text.data(), (int32_t) f.text.size(),
                           tokens.data(), (int32_t) tokens.size(),
                           /* add_special   = */ false,
                           /* parse_special = */ false);
        benchmark::DoNotOptimize(n);
        benchmark::DoNotOptimize(tokens.data());
    }

    if (n < 0) {
        fprintf(stderr, "bench-tokenizer: token buffer too small\n");
        std::abort();
    }
    state.SetLabel(std::to_string(n) + " tokens / " + std::to_string(f.text.size()) + " bytes");
    state.SetBytesProcessed((int64_t) state.iterations() * (int64_t) f.text.size());
}
BENCHMARK(BM_tokenize_qwen35);

BENCHMARK_MAIN();
