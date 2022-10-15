//
// Created by TYTY on 2022-10-09 009.
//

#include <chrono>
#include <fstream>
#include <memory>
#include <string>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#include "layers.h"
#include "md_view.h"
#include "gtest/gtest.h"

#define TEST_PATH "./module/"

#define CUDA_CHECK(status) ASSERT_EQ(status, cudaSuccess)
#define COND_CHECK(cond, message) ASSERT_TRUE(cond)

// small enough to not cause pixel drift
constexpr double Epsilon = 0.5 / 255;

// increase allowed epsilon for fp16.
constexpr double EpsilonHalf = 0.025;

using F = float;
constexpr int32_t input_height = 92;
constexpr int32_t input_width = 160;
constexpr size_t input_count = input_height * input_width;
constexpr uint32_t feature_count = 64;
constexpr size_t input_feature = input_count * feature_count;
static_assert(input_height % 4 == 0 && input_width % 4 == 0);
constexpr int32_t batch_extract = 2;
constexpr int32_t batch_fusion = batch_extract - 1;
constexpr int32_t batch_cycle = 1;

typedef std::chrono::duration<double, std::ratio<1, 1000>> millisecond;

nvinfer1::IRuntime *runtime;
static Logger gLogger;

void loadModel(nvinfer1::IRuntime *runtime, const std::string &path, nvinfer1::ICudaEngine *&engine) {
  std::ifstream file(path, std::ios::binary);
  COND_CHECK(file.good(), "can't open engine file.");

  file.seekg(0, std::ifstream::end);
  auto size = file.tellg();
  file.seekg(0, std::ifstream::beg);
  auto modelStream = std::make_unique<char[]>(size);
  COND_CHECK(modelStream, "Alloc " << size << " bytes failed.");
  file.read(modelStream.get(), size);
  file.close();

  engine = runtime->deserializeCudaEngine(modelStream.get(), size);
}

template<class T, size_t N>
void loadFile(const std::string &path, md_view<T, N> &data) {
  std::ifstream file(path, std::ios::binary);
  COND_CHECK(file.good(), "can't open input file.");

  auto size = data.size() * sizeof(T);
  file.seekg(0, std::ifstream::end);
  ASSERT_EQ(file.tellg(), size);
  file.seekg(0, std::ifstream::beg);
  file.read((char *) data.data, size);
  file.close();
}

void *ptr_add(void *b, size_t n) {
  return static_cast<uint8_t *>(b) + n;
}

template<class T, size_t N>
void compare(md_view<T, N> &actual, md_view<T, N> &expect, double epsilon, const std::string &path) {
  ASSERT_EQ(actual.shape, expect.shape) << path << ": shape mismatch";

  float max = 0;
  double total = 0;

  for (offset_t i = 0; i < actual.size(); ++i) {
    std::stringstream buf;
    buf << '[';
    for (auto idx: actual.shape.indexes(i)) {
      buf << idx << ",";
    }
    buf << "\x08";
    buf << "]";
    EXPECT_NEAR(actual.data[i], expect.data[i], epsilon) << path << ": mismatch happened at " << buf.str();

    float diff = std::abs(actual.data[i] - expect.data[i]);
    total += diff;
    max = diff > max ? diff : max;
  }

  std::cerr << "Diff: max " << max << ", avg " << total / double(actual.size()) << std::endl;
}

TEST(ModelTest, FeatureExtract) {
  ASSERT_NE(runtime, nullptr);

  nvinfer1::ICudaEngine *engine;
  loadModel(runtime, "../models/feature_extract.engine", engine);
  COND_CHECK(engine, "failed deserializing engine");

  auto context = engine->createExecutionContext();
  COND_CHECK(context, "failed creating context");

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  ASSERT_TRUE(context->setOptimizationProfileAsync(0, stream));
  ASSERT_TRUE(context->setBindingDimensions(0, {4, {batch_extract, input_height, input_width, 3}}));

  auto &extract = context->getEngine();

  std::array inputs{
      md_view<F, 4>{nullptr, {batch_extract, input_height, input_width, 3}},
  };

  std::array outputs{
      md_view<F, 4>{nullptr, {batch_extract, input_height, input_width, feature_count}},
      md_view<F, 4>{nullptr, {batch_extract, input_height / 2, input_width / 2, feature_count}},
      md_view<F, 4>{nullptr, {batch_extract, input_height / 4, input_width / 4, feature_count}},
  };

  auto input_buffer = std::make_unique<F[]>(inputs[0].size());
  inputs[0].data = input_buffer.get();
  loadFile(TEST_PATH "fe_i0.bin", inputs[0]);

  auto output_buffer = std::make_unique<F[]>(outputs[0].size() + outputs[1].size() + outputs[2].size());
  outputs[0].data = output_buffer.get();
  outputs[1].data = output_buffer.get() + outputs[0].size();
  outputs[2].data = output_buffer.get() + outputs[0].size() + outputs[1].size();

  void *extractBindings[4] = {};
  void **extractInput = extractBindings;
  void **extractOutput = extractBindings + 1;
  constexpr size_t eSize = sizeof(F);
  CUDA_CHECK(cudaMallocAsync(&extractInput[0], inputs[0].size() * eSize, stream));
  CUDA_CHECK(cudaMallocAsync(&extractOutput[0], outputs[0].size() * eSize, stream));
  CUDA_CHECK(cudaMallocAsync(&extractOutput[1], outputs[1].size() * eSize, stream));
  CUDA_CHECK(cudaMallocAsync(&extractOutput[2], outputs[2].size() * eSize, stream));

  CUDA_CHECK(cudaMemcpyAsync(extractInput[0], inputs[0].data, inputs[0].size() * eSize, cudaMemcpyHostToDevice, stream));

  COND_CHECK(context->enqueueV2(extractBindings, stream, nullptr), "Failed enqueue");

  CUDA_CHECK(cudaMemcpyAsync(outputs[0].data, extractOutput[0], outputs[0].size() * eSize, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(outputs[1].data, extractOutput[1], outputs[1].size() * eSize, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(outputs[2].data, extractOutput[2], outputs[2].size() * eSize, cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::array references{
      md_view<F, 4>{nullptr, {batch_extract, input_height, input_width, feature_count}},
      md_view<F, 4>{nullptr, {batch_extract, input_height / 2, input_width / 2, feature_count}},
      md_view<F, 4>{nullptr, {batch_extract, input_height / 4, input_width / 4, feature_count}},
  };

  auto reference_buffer = std::make_unique<F[]>(references[0].size() + references[1].size() + references[2].size());
  references[0].data = reference_buffer.get();
  references[1].data = reference_buffer.get() + references[0].size();
  references[2].data = reference_buffer.get() + references[0].size() + references[1].size();

  loadFile(TEST_PATH "fe_o0.bin", references[0]);
  loadFile(TEST_PATH "fe_o1.bin", references[1]);
  loadFile(TEST_PATH "fe_o2.bin", references[2]);

  compare(outputs[0], references[0], eSize == 2 ? EpsilonHalf : Epsilon, "output0");
  compare(outputs[1], references[1], eSize == 2 ? EpsilonHalf : Epsilon, "output1");
  compare(outputs[2], references[2], eSize == 2 ? EpsilonHalf : Epsilon, "output2");
}

TEST(ModelTest, FeatureFusion) {
  ASSERT_NE(runtime, nullptr);

  nvinfer1::ICudaEngine *engine;
  loadModel(runtime, "../models/feature_fusion.engine", engine);
  COND_CHECK(engine, "failed deserializing engine");

  auto context = engine->createExecutionContext();
  COND_CHECK(context, "failed creating context");

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  ASSERT_TRUE(context->setOptimizationProfileAsync(0, stream));
  ASSERT_TRUE(context->setBindingDimensions(0, {4, {batch_fusion, feature_count, input_height, input_width}}));
  ASSERT_TRUE(context->setBindingDimensions(1, {4, {batch_fusion, feature_count, input_height / 2, input_width / 2}}));
  ASSERT_TRUE(context->setBindingDimensions(2, {4, {batch_fusion, feature_count, input_height / 4, input_width / 4}}));
  ASSERT_TRUE(context->setBindingDimensions(3, {4, {batch_fusion, feature_count, input_height, input_width}}));
  ASSERT_TRUE(context->setBindingDimensions(4, {4, {batch_fusion, feature_count, input_height / 2, input_width / 2}}));
  ASSERT_TRUE(context->setBindingDimensions(5, {4, {batch_fusion, feature_count, input_height / 4, input_width / 4}}));

  auto &extract = context->getEngine();

  std::array inputs{
      md_view<F, 4>{nullptr, {batch_fusion + 1, input_height, input_width, feature_count}},
      md_view<F, 4>{nullptr, {batch_fusion + 1, input_height / 2, input_width / 2, feature_count}},
      md_view<F, 4>{nullptr, {batch_fusion + 1, input_height / 4, input_width / 4, feature_count}},
  };

  std::array outputs{
      md_view<F, 4>{nullptr, {batch_fusion, input_height, input_width, feature_count}},
  };

  auto input_buffer = std::make_unique<F[]>(inputs[0].size() + inputs[1].size() + inputs[2].size());
  inputs[0].data = input_buffer.get();
  inputs[1].data = input_buffer.get() + inputs[0].size();
  inputs[2].data = input_buffer.get() + inputs[0].size() + inputs[1].size();
  loadFile(TEST_PATH "fe_o0.bin", inputs[0]);
  loadFile(TEST_PATH "fe_o1.bin", inputs[1]);
  loadFile(TEST_PATH "fe_o2.bin", inputs[2]);

  auto output_buffer = std::make_unique<F[]>(outputs[0].size());
  outputs[0].data = output_buffer.get();

  void *fusionBindings[7] = {};
  void **fusionInput = fusionBindings;
  void **fusionOutput = fusionBindings + 6;
  constexpr size_t eSize = sizeof(F);
  CUDA_CHECK(cudaMallocAsync(&fusionInput[0], inputs[0].size() * eSize, stream));
  CUDA_CHECK(cudaMallocAsync(&fusionInput[1], inputs[1].size() * eSize, stream));
  CUDA_CHECK(cudaMallocAsync(&fusionInput[2], inputs[2].size() * eSize, stream));
  fusionInput[3] = ptr_add(fusionInput[0], inputs[0].at(0).size() * eSize),
  fusionInput[4] = ptr_add(fusionInput[1], inputs[1].at(0).size() * eSize),
  fusionInput[5] = ptr_add(fusionInput[2], inputs[2].at(0).size() * eSize);
  CUDA_CHECK(cudaMallocAsync(&fusionOutput[0], outputs[0].size() * eSize, stream));

  CUDA_CHECK(cudaMemcpyAsync(fusionInput[0], inputs[0].data, inputs[0].size() * eSize, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(fusionInput[1], inputs[1].data, inputs[1].size() * eSize, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(fusionInput[2], inputs[2].data, inputs[2].size() * eSize, cudaMemcpyHostToDevice, stream));

  COND_CHECK(context->enqueueV2(fusionBindings, stream, nullptr), "Failed enqueue");

  CUDA_CHECK(cudaMemcpyAsync(outputs[0].data, fusionOutput[0], outputs[0].size() * eSize, cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::array references{
      md_view<F, 4>{nullptr, {batch_fusion, input_height, input_width, feature_count}},
  };

  auto reference_buffer = std::make_unique<F[]>(references[0].size());
  references[0].data = reference_buffer.get();

  loadFile(TEST_PATH "ff_o0.bin", references[0]);

  compare(outputs[0], references[0], sizeof(F) == 2 ? EpsilonHalf : Epsilon, "output0");
}

TEST(ModelTest, MutualCycle) {
  ASSERT_NE(runtime, nullptr);

  nvinfer1::ICudaEngine *engine;
  loadModel(runtime, "../models/mutual_cycle.engine", engine);
  COND_CHECK(engine, "failed deserializing engine");

  auto context = engine->createExecutionContext();
  COND_CHECK(context, "failed creating context");

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  ASSERT_TRUE(context->setOptimizationProfileAsync(0, stream));
  ASSERT_TRUE(context->setBindingDimensions(0, {4, {batch_cycle, feature_count, input_height, input_width}}));
  ASSERT_TRUE(context->setBindingDimensions(1, {4, {batch_cycle, feature_count, input_height, input_width}}));
  ASSERT_TRUE(context->setBindingDimensions(2, {4, {batch_cycle, feature_count, input_height, input_width}}));

  auto &extract = context->getEngine();

  std::array inputs{
      md_view<F, 4>{nullptr, {batch_extract, feature_count, input_height, input_width}},
      md_view<F, 4>{nullptr, {batch_cycle, feature_count, input_height, input_width}},
  };

  std::array outputs{
      md_view<F, 4>{nullptr, {batch_cycle, input_height * 4, input_width * 4, 3}},
      md_view<F, 4>{nullptr, {batch_cycle, input_height * 4, input_width * 4, 3}},
  };

  auto input_buffer = std::make_unique<F[]>(inputs[0].size() * 3);
  inputs[0].data = input_buffer.get();
  inputs[1].data = input_buffer.get() + inputs[0].size();
  loadFile(TEST_PATH "fe_o0.bin", inputs[0]);
  loadFile(TEST_PATH "ff_o0.bin", inputs[1]);

  auto output_buffer = std::make_unique<F[]>(outputs[0].size() + outputs[1].size());
  outputs[0].data = output_buffer.get();
  outputs[1].data = output_buffer.get() + outputs[0].size();

  void *cycleBindings[5] = {};
  void **cycleInput = cycleBindings;
  void **cycleOutput = cycleBindings + 3;
  constexpr size_t eSize = sizeof(F);
  CUDA_CHECK(cudaMallocAsync(&cycleInput[0], inputs[0].size() * eSize, stream));
  CUDA_CHECK(cudaMallocAsync(&cycleInput[1], inputs[1].size() * eSize, stream));
  cycleInput[2] = ptr_add(cycleInput[0], inputs[0].at(0).size() * eSize);
  CUDA_CHECK(cudaMallocAsync(&cycleOutput[0], outputs[0].size() * eSize, stream));
  CUDA_CHECK(cudaMallocAsync(&cycleOutput[1], outputs[1].size() * eSize, stream));

  CUDA_CHECK(cudaMemcpyAsync(cycleInput[0], inputs[0].data, inputs[0].size() * eSize, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(cycleInput[1], inputs[1].data, inputs[1].size() * eSize, cudaMemcpyHostToDevice, stream));

  COND_CHECK(context->enqueueV2(cycleBindings, stream, nullptr), "Failed enqueue");

  CUDA_CHECK(cudaMemcpyAsync(outputs[0].data, cycleOutput[0], outputs[0].size() * eSize, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(outputs[1].data, cycleOutput[1], outputs[1].size() * eSize, cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::array references{
      md_view<F, 4>{nullptr, {batch_cycle, input_height * 4, input_width * 4, 3}},
      md_view<F, 4>{nullptr, {batch_cycle, input_height * 4, input_width * 4, 3}},
  };

  auto reference_buffer = std::make_unique<F[]>(references[0].size() + references[1].size());
  references[0].data = reference_buffer.get();
  references[1].data = reference_buffer.get() + references[0].size();

  loadFile(TEST_PATH "mu_o0.bin", references[0]);
  loadFile(TEST_PATH "mu_o1.bin", references[1]);

  compare(outputs[0], references[0], eSize == 2 ? EpsilonHalf : Epsilon, "output0");
  compare(outputs[1], references[1], eSize == 2 ? EpsilonHalf : Epsilon, "output1");
}

GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from %s\n", __FILE__);
  testing::InitGoogleTest(&argc, argv);

  UDOLayers::registerPlugins();
  runtime = nvinfer1::createInferRuntime(gLogger);

  return RUN_ALL_TESTS();
}
