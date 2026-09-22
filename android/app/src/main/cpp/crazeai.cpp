#include <jni.h>
#include <string>

extern "C" JNIEXPORT jstring JNICALL
Java_ai_craze_runtime_MainActivity_generate(JNIEnv* env, jobject,
                                             jstring prompt, jint, jfloat) {
    const char* p = env->GetStringUTFChars(prompt, nullptr);
    std::string input = p ? p : "";
    env->ReleaseStringUTFChars(prompt, p);

    std::string result =
        "CrazeAI native runtime is ready.\n"
        "Prompt: " + input + "\n"
        "A converted .craze checkpoint is required for model inference.";
    return env->NewStringUTF(result.c_str());
}
