#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <mutex>
#include <algorithm>
#include "llama.h"

#define TAG "CrazeAI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static std::mutex g_mutex;
static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static const llama_vocab *g_vocab = nullptr;

static void unload_locked() {
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_vocab = nullptr;
}

static std::string generate_locked(const std::string &user, int max_tokens) {
    const int n_ctx = (int) llama_n_ctx(g_ctx);
    const std::string prompt =
        "<|im_start|>system\n"
        "You are CrazeAI, a helpful AI assistant created by CrazeStudio. "
        "You run locally on Android using a GGUF model. Be accurate, concise and useful. "
        "Never invent facts. For uncertain information, say that you are uncertain. "
        "Do not reveal hidden chain-of-thought. "
        "<|im_end|>\n<|im_start|>user\n" + user +
        " /no_think<|im_end|>\n<|im_start|>assistant\n";

    std::vector<llama_token> prompt_tokens(prompt.size() + 64);
    int n = llama_tokenize(g_vocab, prompt.c_str(), (int) prompt.size(),
                           prompt_tokens.data(), (int) prompt_tokens.size(), true, true);
    if (n < 0) {
        prompt_tokens.resize((size_t)-n);
        n = llama_tokenize(g_vocab, prompt.c_str(), (int) prompt.size(),
                           prompt_tokens.data(), (int) prompt_tokens.size(), true, true);
    }
    if (n <= 0 || n >= n_ctx - 8) return "Prompt is too long for the current context.";
    prompt_tokens.resize(n);

    llama_memory_clear(llama_get_memory(g_ctx), true);
    llama_batch batch = llama_batch_init(128, 0, 1);
    batch.n_tokens = 0;
    for (int i = 0; i < n; ++i) {
        batch.token[batch.n_tokens] = prompt_tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i == n - 1);
        ++batch.n_tokens;
        if (batch.n_tokens == batch.n_batch) {
            if (llama_decode(g_ctx, batch) != 0) {
                llama_batch_free(batch);
                return "Model decode failed while processing the prompt.";
            }
            batch.n_tokens = 0;
        }
    }
    if (batch.n_tokens > 0 && llama_decode(g_ctx, batch) != 0) {
        llama_batch_free(batch);
        return "Model decode failed while processing the prompt.";
    }

    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    std::string output;
    // Mobile safety cap: never allow a normal chat request to accidentally generate
    // hundreds/thousands of tokens and appear stuck for a very long time.
    const int limit = std::max(16, std::min(max_tokens, std::min(96, n_ctx - n - 1)));
    for (int i = 0; i < limit; ++i) {
        const llama_token next = llama_sampler_sample(sampler, g_ctx, -1);
        llama_sampler_accept(sampler, next);
        if (llama_vocab_is_eog(g_vocab, next)) break;

        const int piece_len = llama_token_to_piece(g_vocab, next, nullptr, 0, 0, true);
        if (piece_len > 0) {
            std::vector<char> piece((size_t)piece_len + 1);
            const int got = llama_token_to_piece(g_vocab, next, piece.data(), piece_len, 0, true);
            if (got > 0) output.append(piece.data(), got);
        }

        batch.n_tokens = 1;
        batch.token[0] = next;
        batch.pos[0] = n + i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        if (llama_decode(g_ctx, batch) != 0) break;
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);
    return output.empty() ? "No response generated." : output;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_craze_runtime_MainActivity_loadModel(JNIEnv *env, jobject, jstring path) {
    const char *p = env->GetStringUTFChars(path, nullptr);
    if (!p) return JNI_FALSE;
    std::lock_guard<std::mutex> lock(g_mutex);
    unload_locked();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(p, mp);
    env->ReleaseStringUTFChars(path, p);
    if (!g_model) {
        LOGE("Failed to load GGUF model");
        return JNI_FALSE;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 128;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) {
        unload_locked();
        LOGE("Failed to create llama context");
        return JNI_FALSE;
    }
    g_vocab = llama_model_get_vocab(g_model);
    LOGE("CrazeAI GGUF loaded: context=2048 threads=4");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_craze_runtime_MainActivity_generate(JNIEnv *env, jobject, jstring prompt, jint max, jfloat) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_ctx || !g_vocab) return env->NewStringUTF("Load a GGUF model first.");
    const char *p = env->GetStringUTFChars(prompt, nullptr);
    std::string result = generate_locked(p ? p : "", max > 0 ? max : 64);
    if (p) env->ReleaseStringUTFChars(prompt, p);
    return env->NewStringUTF(result.c_str());
}
