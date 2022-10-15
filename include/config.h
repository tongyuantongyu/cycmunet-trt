#pragma once

#include <cstdint>

#include <NvInfer.h>

#include "utils.h"

struct optimization_axis {
  optimization_axis(int32_t min, int32_t opt, int32_t max) : min(min), opt(opt), max(max) {}
  optimization_axis(int32_t same) : min(same), opt(same), max(same) {}
  optimization_axis() : min(0), opt(0), max(0) {}
  int32_t min, opt, max;
};

struct OptimizationConfig {
  optimization_axis input_width;
  optimization_axis input_height;

  optimization_axis batch_extract;
  optimization_axis batch_fusion;
  optimization_axis batch_cycle;

  int32_t feature_count;
  int32_t deformable_groups;

  bool use_fp16;
};

struct InferenceConfig {
  int32_t input_width;
  int32_t input_height;

  int32_t batch_extract;
  int32_t batch_fusion;
  int32_t batch_cycle;

  int32_t feature_count;

  bool use_fp16;
};
