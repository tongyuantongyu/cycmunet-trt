//
// Created by TYTY on 2021-12-23 023.
//

#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "logging.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

#include "image-utils.h"
#include "layers.h"

#define CUDA_CHECK(status) \
    do \
    { \
        auto ret = (status); \
        if (ret != 0) \
        { \
            std::cerr << "Cuda failure at " __FILE__ ":" << __LINE__ << ": " << ret << std::endl; \
            exit(2); \
        } \
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

void copyAreaImport(const uint8_t *src, size_t s_stride, pixel *dst, size_t d_stride, size_t w, size_t h) {
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      dst[d_stride * y + x] = float(src[s_stride * y + x]) / 255;
    }
  }
}

void copyAreaExport(const pixel *src, size_t s_stride, uint8_t *dst, size_t d_stride, size_t w, size_t h) {
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      auto p = src[s_stride * y + x];
      p = p < 0 ? 0 : (p > 1 ? 1 : p);
      dst[d_stride * y + x] = uint8_t(std::round(p * 255));
    }
  }
}

void copyAreaSplit(const pixel *src, size_t s_stride, pixel *dst, size_t d_stride, size_t d_gap, size_t w, size_t h) {
  for (size_t y = 0; y < h; ++y) {
    const pixel *srcLine = src + s_stride * y;
    pixel *dstLine = dst + d_stride * y;
    for (size_t x = 0; x < w; ++x) {
      dstLine[x + 0 * d_gap] = srcLine[3 * x + 0];
      dstLine[x + 1 * d_gap] = srcLine[3 * x + 1];
      dstLine[x + 2 * d_gap] = srcLine[3 * x + 2];
    }
  }
}

void copyAreaMerge(const pixel *src, size_t s_stride, size_t s_gap, pixel *dst, size_t d_stride, size_t w, size_t h) {
  for (size_t y = 0; y < h; ++y) {
    const pixel *srcLine = src + s_stride * y;
    pixel *dstLine = dst + d_stride * y;
    for (size_t x = 0; x < w; ++x) {
      dstLine[3 * x + 0] = srcLine[x + 0 * s_gap];
      dstLine[3 * x + 1] = srcLine[x + 1 * s_gap];
      dstLine[3 * x + 2] = srcLine[x + 2 * s_gap];
    }
  }
}

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

void loadFile(const std::string& path, char* data, size_t size) {
  std::ifstream file(path, std::ios::binary);
  COND_CHECK(file.good(), "can't open input file: " << path);

  file.read(data, size);
  file.close();
}

void saveFile(const std::string& path, const char* data, size_t size) {
  std::ofstream file(path, std::ios::binary);
  COND_CHECK(file.good(), "can't open output file: " << path);

  file.write(data, size);
  file.close();
}

void saveFileNv(const std::string& path, const void* data, size_t size) {
  std::ofstream file(path, std::ios::binary);
  COND_CHECK(file.good(), "can't open output file: " << path);

  auto tmp = std::make_unique<char[]>(size);
  CUDA_CHECK(cudaMemcpy(tmp.get(), data, size, cudaMemcpyDeviceToHost));

  file.write(tmp.get(), size);
  file.close();
}

template<class T>
struct ModelStuff {
  T* feature_extract;
  T* feature_fusion;
  T* mutual_cycle;
};

template<class T>
ModelStuff(T*) -> ModelStuff<T>;

template<class T>
ModelStuff(T*, T*) -> ModelStuff<T>;

template<class T>
ModelStuff(T*, T*, T*) -> ModelStuff<T>;

template<class T, class U, class V, std::enable_if_t<std::is_convertible_v<U*, T*> && std::is_convertible_v<V*, T*>, bool> = true>
ModelStuff(T*, U*, V*) -> ModelStuff<T>;

void* ptr_add(void* b, size_t n) {
  return static_cast<uint8_t*>(b) + n;
}

typedef std::chrono::duration<double, std::ratio<1, 1000>> millisecond;

constexpr bool use_fp16 = true;

int main() {
//  initImageTool();
  UDOLayers::registerPlugins();

  auto runtime = nvinfer1::createInferRuntime(gLogger);
  COND_CHECK(runtime, "failed creating runtime");

  ModelStuff engine {
      loadModel(runtime, "models/feature_extract.engine"),
      loadModel(runtime, "models/feature_fusion.engine"),
      loadModel(runtime, "models/mutual_cycle.engine")
  };

  std::cerr << engine.feature_extract->getDeviceMemorySize() << " "
            << engine.feature_fusion->getDeviceMemorySize() << " "
            << engine.mutual_cycle->getDeviceMemorySize() << std::endl;

  ModelStuff context {
      engine.feature_extract->createExecutionContext(),
      engine.feature_fusion->createExecutionContext(),
      engine.mutual_cycle->createExecutionContext(),
  };

  ModelStuff inspector {
      engine.feature_extract->createEngineInspector(),
      engine.feature_fusion->createEngineInspector(),
      engine.mutual_cycle->createEngineInspector(),
  };

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // old result
  // 256x128:  6107MB, 161ms
  // 320x180:  9877MB, 276ms
  // 320x240: 12901MB, 365ms
  // 360x240: 14205MB, 406ms
  // 380x240: 15801MB, 432ms
  // 360x256, 15929MB, 440ms

  // new result
  // 320x180,    8469MB, 230ms
  // 360x240:   12139MB, 342ms
  // 480x272,   17797MB, 505ms?
  // 320x180x3, 19413MB, 668ms

  // new+lowmem result
  // 320x180,    6267MB
  // 360x240,    8793MB
  // 480x272,   12883MB
  // 320x180x3, 12805MB
  // 320x180x4, 16179MB
  // 640x360,   21701MB, 907ms

  constexpr int32_t input_height = 180;
  constexpr int32_t input_width = 320;
  constexpr size_t input_count = input_height * input_width;
  constexpr uint32_t feature_count = 64;
  constexpr size_t input_feature = input_count * feature_count;
  static_assert(input_height % 4 == 0 && input_width % 4 == 0);
  constexpr int32_t batch_extract = 5;
  constexpr int32_t batch_fusion = batch_extract - 1;
  constexpr int32_t batch_cycle = batch_fusion;

  context.feature_extract->setOptimizationProfileAsync(0, stream);
  context.feature_extract->setBindingDimensions(0, {4, {batch_extract, input_height, input_width, 3}});
  inspector.feature_extract->setExecutionContext(context.feature_extract);

  context.feature_fusion->setOptimizationProfileAsync(0, stream);
  context.feature_fusion->setBindingDimensions(0, {4, {batch_fusion, feature_count, input_height, input_width}});
  context.feature_fusion->setBindingDimensions(1, {4, {batch_fusion, feature_count, input_height / 2, input_width / 2}});
  context.feature_fusion->setBindingDimensions(2, {4, {batch_fusion, feature_count, input_height / 4, input_width / 4}});
  context.feature_fusion->setBindingDimensions(3, {4, {batch_fusion, feature_count, input_height, input_width}});
  context.feature_fusion->setBindingDimensions(4, {4, {batch_fusion, feature_count, input_height / 2, input_width / 2}});
  context.feature_fusion->setBindingDimensions(5, {4, {batch_fusion, feature_count, input_height / 4, input_width / 4}});
  inspector.feature_fusion->setExecutionContext(context.feature_fusion);

  context.mutual_cycle->setOptimizationProfileAsync(0, stream);
  context.mutual_cycle->setBindingDimensions(0, {4, {batch_cycle, feature_count, input_height, input_width}});
  context.mutual_cycle->setBindingDimensions(1, {4, {batch_cycle, feature_count, input_height, input_width}});
  context.mutual_cycle->setBindingDimensions(2, {4, {batch_cycle, feature_count, input_height, input_width}});
  inspector.mutual_cycle->setExecutionContext(context.mutual_cycle);

  cudaStreamSynchronize(stream);

  auto& extract = context.feature_extract->getEngine();
  auto& fusion = context.feature_fusion->getEngine();

  void *extractBindings[4] = {};
  void **extractInput = extractBindings;
  void **extractOutput = extractBindings + 1;
  constexpr size_t eSize = use_fp16 ? 2 : 4;
  CUDA_CHECK(cudaMallocAsync(&extractInput[0], batch_extract * input_count * 3 * eSize, stream));
  CUDA_CHECK(cudaMallocAsync(&extractOutput[0], batch_extract * input_feature * eSize, stream));
  CUDA_CHECK(cudaMallocAsync(&extractOutput[1], batch_extract * input_feature / 4 * eSize, stream));
  CUDA_CHECK(cudaMallocAsync(&extractOutput[2], batch_extract * input_feature / 16 * eSize, stream));

  void *fusionBindings[7] = {
      extractOutput[0],
      extractOutput[1],
      extractOutput[2],
      ptr_add(extractOutput[0], input_feature * eSize),
      ptr_add(extractOutput[1], input_feature / 4 * eSize),
      ptr_add(extractOutput[2], input_feature / 16 * eSize),
  };
  void **fusionOutput = fusionBindings + 6;
  CUDA_CHECK(cudaMallocAsync(&fusionOutput[0], batch_fusion * input_feature * eSize, stream));

  void* cycleBindings[5] = {
      fusionBindings[0],
      fusionOutput[0],
      fusionBindings[3]
  };
  void **cycleOutput = cycleBindings + 3;
  CUDA_CHECK(cudaMallocAsync(&cycleOutput[0], 2 * batch_cycle * input_count * 16 * 3 * eSize, stream));
  cycleOutput[1] = ptr_add(cycleOutput[0], batch_cycle * input_count * 16 * 3 * eSize);

//  cudaStreamSynchronize(stream);

  std::cout << "Initialization done." << std::endl;

  auto input_size = batch_extract * input_count * 3 * eSize;
  auto output_size = 2 * batch_cycle * input_count * 16 * 3 * eSize;
  auto host_buffer_in = std::make_unique<char[]>(input_size);
  auto host_buffer_out = std::make_unique<char[]>(output_size);

  for (int i = 1; i < 249; i += 8) {
    auto digit = std::to_string(i);
    digit = std::string(3 - digit.size(), '0') + digit;

  loadFile("src_raw/" + digit + ".bin", host_buffer_in.get(), input_size);

  auto all_begin = std::chrono::steady_clock::now();
  millisecond elapsed;

  CUDA_CHECK(cudaMemcpyAsync(extractInput[0], host_buffer_in.get(), input_size, cudaMemcpyHostToDevice, stream));

  context.feature_extract->enqueueV2(extractBindings, stream, nullptr);
//    cudaStreamSynchronize(stream);
//    elapsed = std::chrono::steady_clock::now() - all_begin;
//    std::cerr << "feature_extract done after " << elapsed.count() << "ms.\n";
//  cudaStreamSynchronize(stream);
//  saveFileNv("dst_raw/fe_l1.bin", extractBindings[1], batch_extract * input_feature * eSize);
//  saveFileNv("dst_raw/fe_l2.bin", extractBindings[2], batch_extract * input_feature * eSize / 4);
//  saveFileNv("dst_raw/fe_l3.bin", extractBindings[3], batch_extract * input_feature * eSize / 16);

  context.feature_fusion->enqueueV2(fusionBindings, stream, nullptr);
//    cudaStreamSynchronize(stream);
//    elapsed = std::chrono::steady_clock::now() - all_begin;
//    std::cerr << "feature_fusion done after " << elapsed.count() << "ms.\n";
//  cudaStreamSynchronize(stream);
//  saveFileNv("dst_raw/ff.bin", fusionBindings[6], batch_fusion * input_feature * eSize);

  context.mutual_cycle->enqueueV2(cycleBindings, stream, nullptr);

  CUDA_CHECK(cudaMemcpyAsync(host_buffer_out.get(), cycleOutput[0], output_size, cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
  elapsed = std::chrono::steady_clock::now() - all_begin;

  std::cerr << "mutual_cycle #" << i << " done after " << elapsed.count() << "ms." << std::endl << std::endl;

  saveFile("dst_raw/" + digit + ".bin", host_buffer_out.get(), output_size);
  }
}