#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <mutex>
#include "llama.h"

#define TAG "CrazeAI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static std::mutex g_mutex;
static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static const llama_vocab *g_vocab = nullptr;

static void unload_locked() {
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_vocab = nullptr;
}

static std::string utf8_from_tokens(const std::vector<llama_token> &tokens) {
    std::string out;
    char buf[256];
    for (llama_token token : tokens) {
        int n = llama_token_to_piece(g_vocab, token, buf, sizeof(buf), 0, true);
        if (n > 0) out.append(buf, n);
    }
    return out;
}

static std::string generate_locked(const std::string &user, int max_tokens) {
    const int n_ctx = (int) llama_n_ctx(g_ctx);
    const std::string prompt =
        "<|im_start|>system\n"
        "You are CrazeAI, an AI assistant created by CrazeStudio. "
        "Your underlying base model is Qwen3-1.7B. Be accurate, useful and concise. "
        "Do not invent facts when uncertain.<|im_end|>\n"
        "<|im_start|>user\n" + user +
        "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<llama_token> prompt_tokens(prompt.size() + 32);
    int n = llama_tokenize(g_vocab, prompt.c_str(), (int) prompt.size(),
                           prompt_tokens.data(), (int) prompt_tokens.size(), true, true);
    if (n < 0) {
        prompt_tokens.resize(-n);
        n = llama_tokenize(g_vocab, prompt.c_str(), (int) prompt.size(),
                           prompt_tokens.data(), (int) prompt_tokens.size(), true, true);
    }
    if (n <= 0 || n >= n_ctx - 4) return "Prompt is too long for this model context.";
    prompt_tokens.resize(n);

    llama_memory_clear(llama_get_memory(g_ctx), true);
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    batch.n_tokens = 0;
    for (int i = 0; i < n; ++i) {
        batch.token[batch.n_tokens] = prompt_tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i == n - 1);
        ++batch.n_tokens;
    }

    if (llama_decode(g_ctx, batch) != 0) {
        llama_batch_free(batch);
        return "Model decode failed.";
    }

    std::string output;
    const int limit = std::min(max_tokens, n_ctx - n - 1);
    for (int i = 0; i < limit; ++i) {
        float * logits = llama_get_logits(g_ctx);
        const int vocab_size = llama_vocab_n_tokens(g_vocab);
        llama_token next = 0;
        float best = -1e30f;
        for (int t = 0; t < vocab_size; ++t) {
            if (logits[t] > best) {
                best = logits[t];
                next = t;
            }
        }

        if (llama_vocab_is_eog(g_vocab, next)) break;
        const int piece_len = llama_token_to_piece(g_vocab, next, nullptr, 0, 0, true);
        if (piece_len > 0) {
            std::vector<char> piece((size_t) piece_len + 1);
            int got = llama_token_to_piece(g_vocab, next, piece.data(), piece_len, 0, true);
            if (got > 0) output.append(piece.data(), got);
        }

        batch.n_tokens = 0;
        batch.token[0] = next;
        batch.pos[0] = n + i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        batch.n_tokens = 1;
        if (llama_decode(g_ctx, batch) != 0) break;
    }

    llama_batch_free(batch);
    return output.empty() ? "No response generated." : output;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_craze_runtime_MainActivity_loadModel(JNIEnv *env, jobject, jstring path) {
    const char *p = env->GetStringUTFChars(path, nullptr);
    if (!p) return JNI_FALSE;
    std::lock_guard<std::mutex> lock(g_mutex);
    unload_locked();

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
    cp.n_batch = 512;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) {
        unload_locked();
        LOGE("Failed to create llama context");
        return JNI_FALSE;
    }
    g_vocab = llama_model_get_vocab(g_model);
    LOGE("GGUF model loaded successfully");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_craze_runtime_MainActivity_generate(JNIEnv *env, jobject, jstring prompt, jint max, jfloat) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_ctx || !g_vocab) return env->NewStringUTF("Load a GGUF model first.");
    const char *p = env->GetStringUTFChars(prompt, nullptr);
    std::string result = generate_locked(p ? p : "", max > 0 ? max : 256);
    if (p) env->ReleaseStringUTFChars(prompt, p);
    return env->NewStringUTF(result.c_str());
}
