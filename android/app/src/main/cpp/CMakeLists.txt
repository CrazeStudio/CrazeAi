cmake_minimum_required(VERSION 3.22.1)
project(crazeai)

include(FetchContent)

FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama)

add_library(crazeai SHARED crazeai.cpp)
target_compile_features(crazeai PRIVATE cxx_std_17)

# llama.cpp/ggml requires IEEE non-finite math support. Do not use
# -ffast-math here because Clang enables -ffinite-math-only with it.
target_compile_options(crazeai PRIVATE -O3 -fno-finite-math-only -fomit-frame-pointer)

# FetchContent's ggml CPU targets are compiled separately; apply the same
# required compiler flag to them as well.
target_compile_options(ggml PRIVATE -fno-finite-math-only)
target_compile_options(ggml-base PRIVATE -fno-finite-math-only)
target_compile_options(ggml-cpu PRIVATE -fno-finite-math-only)
target_compile_options(llama PRIVATE -fno-finite-math-only)

target_include_directories(crazeai PRIVATE ${llama_SOURCE_DIR}/include ${llama_SOURCE_DIR}/ggml/include)
target_link_libraries(crazeai PRIVATE llama log)
