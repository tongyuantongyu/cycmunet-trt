cmake_minimum_required(VERSION 3.18)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
        set(CMAKE_CUDA_ARCHITECTURES 53 62 72)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 61 75 80)
    endif()
endif()

project(Cycmunet-TRT LANGUAGES CXX CUDA)

enable_testing()

option(CUDA_DEVICE_DEBUG "Enable device debug" OFF)
if (CUDA_DEVICE_DEBUG)
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
endif()

#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xptxas -v")

if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/source-charset:utf-8 /execution-charset:us-ascii>")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/source-charset:utf-8$<SEMICOLON>/execution-charset:us-ascii>)
endif()

find_package(CUDAToolkit 11.0 REQUIRED COMPONENTS cublas)
find_package(TensorRT 8.2.0 REQUIRED COMPONENTS OnnxParser)
find_package(cuDNN REQUIRED)
find_package(Protobuf 3 REQUIRED)
find_package(GTest 1 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(COMMON_HEADERS include/helper.h include/md_view.h include/utils.h include/logging.h)
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(COMMON_HEADERS ${COMMON_HEADERS} include/debug/reveal.h)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/debug)
endif()

add_library(_dcn_layer_impl OBJECT
        ${COMMON_HEADERS}
        layers/include/internal/config.h

        layers/include/internal/dcn_layer_impl.h
        layers/impl/dcn_layer.cu)
target_include_directories(_dcn_layer_impl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/layers/include/internal)
target_link_libraries(_dcn_layer_impl CUDA::cublas)

add_executable(dcn_layer_test layers/test/dcn_layer_test.cpp)
target_include_directories(dcn_layer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/layers/include/internal)
target_link_libraries(dcn_layer_test _dcn_layer_impl CUDA::cudart GTest::gtest_main)
add_test(NAME dcn_layer_test COMMAND dcn_layer_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/layers/test)

option(AUTO_REGISTER_PLUGIN "Automatically register plugin on load of shared library" ON)
add_library(trt_layers_plugin SHARED
        ${COMMON_HEADERS}
        layers/include/internal/config.h

        layers/include/internal/dcn_layer.h
        layers/include/internal/dcn_layer_impl.h
        layers/src/dcn_layer.cpp

        layers/src/layers.cpp)

target_compile_definitions(trt_layers_plugin PRIVATE BUILDING_PLUGIN)
if (AUTO_REGISTER_PLUGIN)
    target_compile_definitions(trt_layers_plugin PRIVATE AUTO_REGISTER_PLUGIN)
endif()
target_include_directories(trt_layers_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/layers/include/internal)
target_include_directories(trt_layers_plugin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/layers/include)
target_link_libraries(trt_layers_plugin PRIVATE _dcn_layer_impl TensorRT::NvInfer)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS weights.proto)
add_library(weights_pb OBJECT ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(weights_pb protobuf::libprotobuf)
target_include_directories(weights_pb INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_executable(optimizer optimizer.cpp)
target_link_libraries(optimizer PRIVATE trt_layers_plugin weights_pb CUDA::cudart TensorRT::NvInfer TensorRT::OnnxParser)

set(INFERENCE_SOURCES inference.cpp image-utils.h)
if(WIN32)
    set(INFERENCE_SOURCES ${INFERENCE_SOURCES} image-wic.cpp)
endif()
add_executable(inference ${INFERENCE_SOURCES})
target_compile_definitions(inference PRIVATE USE_FP32=1)
target_link_libraries(inference PRIVATE trt_layers_plugin CUDA::cudart TensorRT::NvInfer)
if(WIN32)
    target_link_libraries(inference PRIVATE shlwapi windowscodecs)
endif()

add_executable(model_test test/model_test.cpp)
target_link_libraries(model_test trt_layers_plugin CUDA::cudart TensorRT::NvInfer GTest::gtest)
add_test(NAME model_test COMMAND model_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
