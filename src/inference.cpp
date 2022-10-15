//
// Created by TYTY on 2021-12-23 023.
//

#include <iostream>
#include <fstream>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "logging.h"
#include "inference.h"

#define CUDA_CHECK(status)                                                                                             \
  do {                                                                                                                 \
    auto ret = (status);                                                                                               \
    if (ret != 0) {                                                                                                    \
      std::cerr << "Cuda failure at " __FILE__ ":" << __LINE__ << ": " << ret << std::endl;                            \
      exit(2);                                                                                                         \
    }                                                                                                                  \
  } while (0)

#define COND_CHECK(cond, message) \
    do \
    { \
        if (!(cond)) \
        { \
            std::cerr << "Check failed " __FILE__ ":" << __LINE__ << ": " #cond ", " << message << std::endl; \
            exit(2); \
        } \
    } while (0)

const char *InputFeatureExtract = "frame";
const char *OutputFeatureExtract[] = {"l1", "l2", "l3"};
const char *InputFeatureFusion[] = {"f0l1", "f0l2", "f0l3", "f2l1", "f2l2", "f2l3"};
const char *OutputFeatureFusion = "f1l1";
const char *InputMutualCycle[] = {"lf0", "lf1", "lf2"};
const char *OutputMutualCycle[] = {"h0", "h1", "h2"};

static Logger gLogger;

nvinfer1::ICudaEngine* loadModel(nvinfer1::IRuntime * runtime, const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  COND_CHECK(file.good(), "can't open engine file.");

  file.seekg(0, std::ifstream::end);
  auto size = file.tellg();
  file.seekg(0, std::ifstream::beg);
  auto modelStream = std::make_unique<char[]>(size);
  COND_CHECK(modelStream, "Alloc " << size << " bytes failed.");
  file.read(modelStream.get(), size);
  file.close();

  auto engine = runtime->deserializeCudaEngine(modelStream.get(), size);
  COND_CHECK(runtime, "failed deserializing engine");

  return engine;
}

InferenceContext::InferenceContext(InferenceConfig config, const ModelStuff<std::string> &modelPath)
    : config(config),
      runtime(nvinfer1::createInferRuntime(gLogger)), engine{loadModel(runtime, modelPath.feature_extract),
                                                             loadModel(runtime, modelPath.feature_fusion),
                                                             loadModel(runtime, modelPath.mutual_cycle)} {}

static void *ptr_add(void *b, size_t n) { return static_cast<uint8_t *>(b) + n; }

InferenceSession::InferenceSession(InferenceContext &ctx)
    : ctx(ctx), context{ctx.engine.feature_extract->createExecutionContext(),
                        ctx.engine.feature_fusion->createExecutionContext(),
                        ctx.engine.mutual_cycle->createExecutionContext()} {
  const size_t eSize = ctx.config.use_fp16 ? 2 : 4;
  auto input_height = ctx.config.input_height;
  auto input_width = ctx.config.input_width;
  auto feature_count = ctx.config.feature_count;
  const size_t input_count = input_height * input_width;
  const size_t input_feature = input_count * feature_count;

  void **extractInput = extractBindings.data();
  void **extractOutput = extractBindings.data() + 1;

  CUDA_CHECK(cudaStreamCreate(&stream));

  context.feature_extract->setOptimizationProfileAsync(0, stream);
  context.feature_extract->setBindingDimensions(0, {4, {ctx.config.batch_extract, input_height, input_width, 3}});

  context.feature_fusion->setOptimizationProfileAsync(0, stream);
  context.feature_fusion->setBindingDimensions(
      0, {4, {ctx.config.batch_fusion, feature_count, input_height, input_width}});
  context.feature_fusion->setBindingDimensions(
      1, {4, {ctx.config.batch_fusion, feature_count, input_height / 2, input_width / 2}});
  context.feature_fusion->setBindingDimensions(
      2, {4, {ctx.config.batch_fusion, feature_count, input_height / 4, input_width / 4}});
  context.feature_fusion->setBindingDimensions(
      3, {4, {ctx.config.batch_fusion, feature_count, input_height, input_width}});
  context.feature_fusion->setBindingDimensions(
      4, {4, {ctx.config.batch_fusion, feature_count, input_height / 2, input_width / 2}});
  context.feature_fusion->setBindingDimensions(
      5, {4, {ctx.config.batch_fusion, feature_count, input_height / 4, input_width / 4}});

  context.mutual_cycle->setOptimizationProfileAsync(0, stream);
  context.mutual_cycle->setBindingDimensions(0,
                                             {4, {ctx.config.batch_cycle, feature_count, input_height, input_width}});
  context.mutual_cycle->setBindingDimensions(1,
                                             {4, {ctx.config.batch_cycle, feature_count, input_height, input_width}});
  context.mutual_cycle->setBindingDimensions(2,
                                             {4, {ctx.config.batch_cycle, feature_count, input_height, input_width}});

  CUDA_CHECK(cudaMallocAsync(&extractInput[0], ctx.config.batch_extract * input_count * 3 * eSize, stream));
  CUDA_CHECK(cudaMallocAsync(&extractOutput[0], ctx.config.batch_extract * input_feature * eSize, stream));
  CUDA_CHECK(cudaMallocAsync(&extractOutput[1], ctx.config.batch_extract * input_feature / 4 * eSize, stream));
  CUDA_CHECK(cudaMallocAsync(&extractOutput[2], ctx.config.batch_extract * input_feature / 16 * eSize, stream));

  fusionBindings = {
      extractOutput[0],
      extractOutput[1],
      extractOutput[2],
      ptr_add(extractOutput[0], input_feature * eSize),
      ptr_add(extractOutput[1], input_feature / 4 * eSize),
      ptr_add(extractOutput[2], input_feature / 16 * eSize),
  };

  void **fusionOutput = fusionBindings.data() + 6;
  CUDA_CHECK(cudaMallocAsync(&fusionOutput[0], ctx.config.batch_fusion * input_feature * eSize, stream));

  cycleBindings = {fusionBindings[0], fusionOutput[0], fusionBindings[3]};
  void **cycleOutput = cycleBindings.data() + 3;
  CUDA_CHECK(cudaMallocAsync(&cycleOutput[0], 2 * ctx.config.batch_cycle * input_count * 16 * 3 * eSize, stream));
  cycleOutput[1] = ptr_add(cycleOutput[0], ctx.config.batch_cycle * input_count * 16 * 3 * eSize);

  input[0] = extractInput[0];
  output[0] = cycleOutput[0];
  output[1] = cycleOutput[1];

  input_size = ctx.config.batch_extract * input_count * 3 * eSize;
  output_size = ctx.config.batch_cycle * input_count * 16 * 3 * eSize;
}

bool InferenceSession::enqueue(cudaEvent_t *event) {
  if (!context.feature_extract->enqueueV2(extractBindings.data(), stream, event)) { return false; }

  if (!context.feature_fusion->enqueueV2(fusionBindings.data(), stream, nullptr)) { return false; }

  return context.mutual_cycle->enqueueV2(cycleBindings.data(), stream, nullptr);
}
