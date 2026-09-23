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
target_compile_options(crazeai PRIVATE -O3 -ffast-math -fomit-frame-pointer)

target_include_directories(crazeai PRIVATE ${llama_SOURCE_DIR}/include ${llama_SOURCE_DIR}/ggml/include)
target_link_libraries(crazeai PRIVATE llama log)
