#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string_view>

#include "cuda_runtime_api.h"

#include "inference.h"
#include "layers.h"

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

typedef std::chrono::duration<double, std::ratio<1, 1000>> millisecond;

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

int main(int argc, char** argv) {
  UDOLayers::registerPlugins();

  InferenceConfig config {
      320, 180,
      2, 1, 1,
      64,
      false
  };

  InferenceContext context {
      config,
      {
          "model/feature_extract.engine",
          "model/feature_fusion.engine",
          "model/mutual_cycle.engine"
      }
  };

  InferenceSession session { context };

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

  std::cout << "Initialization done." << std::endl;

  auto host_buffer_in = std::make_unique<char[]>(session.input_size);
  auto host_buffer_out = std::make_unique<char[]>(2 * session.output_size);

  millisecond elapsed;
  for (int i = 1; i < argc; ++i) {
    loadFile(argv[i], host_buffer_in.get(), session.input_size);

    auto begin = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaMemcpyAsync(session.input[0], host_buffer_in.get(), session.input_size, cudaMemcpyHostToDevice, session.stream));
    session.enqueue(nullptr);
    CUDA_CHECK(cudaMemcpyAsync(host_buffer_out.get(), session.output[0], 2 * session.output_size, cudaMemcpyDeviceToHost, session.stream));
    cudaStreamSynchronize(session.stream);
    elapsed = std::chrono::steady_clock::now() - begin;

    std::cerr << "File " << argv[i] << " done after " << elapsed.count() << "ms." << std::endl;

    std::string f(argv[i]);
    auto pos = f.rfind('.');
    auto output = f.substr(0, pos) + "_out.bin";
    saveFile(output, host_buffer_out.get(), 2 * session.output_size);
  }
}