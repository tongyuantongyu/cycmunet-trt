#pragma once

#include <cstdint>
#include <string>

#include <NvInfer.h>

#include "utils.h"
#include "config.h"

template<class T>
struct ModelStuff {
  T feature_extract;
  T feature_fusion;
  T mutual_cycle;
};

class InferenceSession;

class InferenceContext {
  InferenceConfig config;

  nvinfer1::IRuntime* runtime;
  ModelStuff<nvinfer1::ICudaEngine*> engine;

  friend class InferenceSession;

 public:
  InferenceContext(InferenceConfig config, const ModelStuff<std::string>& modelPath);
};

class InferenceSession {
  InferenceContext ctx;

  ModelStuff<nvinfer1::IExecutionContext*> context;
  std::array<void*, 4> extractBindings;
  std::array<void*, 7> fusionBindings;
  std::array<void*, 5> cycleBindings;

 public:
  cudaStream_t stream;
  std::array<void*, 1> input;
  std::array<void*, 2> output;
  size_t input_size;
  size_t output_size;

  explicit InferenceSession(InferenceContext& ctx);

  bool enqueue(cudaEvent_t* event);
};