//
// Created by TYTY on 2021-12-23 023.
//

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"

#include "logging.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "layers.h"
#include "weights.pb.h"

#define CUDA_CHECK(status)                               \
  do {                                                   \
    auto ret = (status);                                 \
    if (ret != 0) {                                      \
      std::cerr << "Cuda failure: " << ret << std::endl; \
      exit(2);                                           \
    }                                                    \
  } while (0)

#define COND_CHECK(cond, message)                                       \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::cerr << "Check failed: " #cond ", " << message << std::endl; \
      exit(2);                                                          \
    }                                                                   \
  } while (0)

class OwnedWeights : public nvinfer1::Weights {
 public:
  explicit OwnedWeights(nvinfer1::DataType type) : nvinfer1::Weights{type, nullptr, 0} {}
  OwnedWeights(nvinfer1::DataType t, const void *v, int64_t c) : nvinfer1::Weights{t, v, c} {}
  OwnedWeights(nvinfer1::DataType t, const uint8_t *v, int64_t c) : nvinfer1::Weights{t, v, c} {}
  OwnedWeights(OwnedWeights &&other) noexcept : nvinfer1::Weights{other.type, other.values, other.count} {
    other.values = nullptr;
    other.count = 0;
  }
  ~OwnedWeights() {
    if (values != nullptr) {
      delete[] reinterpret_cast<const uint8_t *>(values);
    }
  }
};

typedef std::unordered_map<std::string, OwnedWeights> WeightMap;

WeightMap loadWeights(const std::string &file) {
  std::cout << "Loading weights: " << file << std::endl;
  WeightMap weightMap;

  // Open weights file
  std::ifstream input(file, std::ios::binary | std::ios::in);
  COND_CHECK(input.is_open(), "Unable to load weight file.");

  weights::Weights pbWeight;
  COND_CHECK(pbWeight.ParseFromIstream(&input), "Bad weight format");
  for (const auto &[name, value]: pbWeight.weights()) {
    assert(value.size() % 4 == 0);
    auto *values = new uint8_t[value.size()];
    memcpy(values, value.c_str(), value.size());
    auto [_, insert] =
        weightMap.try_emplace(name, OwnedWeights{nvinfer1::DataType::kFLOAT, values, int64_t(value.size() / 4)});
    COND_CHECK(insert, "duplicate layer name: " << name);
  }

  std::cout << "Done loading weights." << std::endl;
  return weightMap;
}

void inspectNetwork(nvinfer1::INetworkDefinition *network) {
  auto count = network->getNbLayers();
  for (int i = 0; i < count; ++i) {
    auto layer = network->getLayer(i);
    auto i_count = layer->getNbInputs();
    std::cout << "#" << i << ": " << layer->getName() << ", " << int32_t(layer->getType()) << ": ";

    std::cout << "from {";
    for (int j = 0; j < i_count; ++j) {
      std::string name = layer->getInput(j)->getName();
      if (name.size() > 15) {
        name = std::to_string(atoi(name.c_str() + 15));
      }
      std::cout << name;
      auto size = layer->getInput(j)->getDimensions();
      std::cout << "(";
      for (int k = 0; k < size.nbDims; ++k) {
        std::cout << size.d[k] << ",";
      }
      std::cout << "\x08), ";
    }
    std::cout << "\x08\x08}\n";
  }
  std::cout.flush();
}

struct NetworkDefinitionHelper {
  nvinfer1::INetworkDefinition *network;
  const WeightMap &weight_map;
  std::vector<std::unique_ptr<float[]>> buffers;
  std::unordered_map<std::string, uint64_t> naming;

  static nvinfer1::IPluginRegistry *plugins;

  template<typename Kernel, typename Stride, typename Padding>
  nvinfer1::ITensor *makeConv2d(const std::string &name, nvinfer1::ITensor *input, int32_t channels, Kernel k, Stride s,
                                Padding p) {
    nvinfer1::DimsHW kernel = asDims(k), stride = asDims(s), padding = asDims(p);
    auto conv = network->addConvolutionNd(*input, channels,
                                          kernel,
                                          weight_map.at(name + ".weight"),
                                          weight_map.at(name + ".bias"));
    conv->setStrideNd(stride);
    conv->setPaddingNd(padding);
    setName(conv, name);
    return conv->getOutput(0);
  }

  template<typename Kernel, typename Stride, typename Padding>
  nvinfer1::ITensor *makeDeformableConv2dAc(const std::string &name,
                                            nvinfer1::ITensor *input,
                                            nvinfer1::ITensor *feature,
                                            int32_t channels,
                                            Kernel k,
                                            Stride s,
                                            Padding p,
                                            int32_t deformable_groups,
                                            nvinfer1::ActivationType type,
                                            float alpha) {
    nvinfer1::DimsHW kernel = asDims(k), stride = asDims(s), padding = asDims(p);

    // this is one conv in original network, separated here for activation fusion. This reduces workspace consumption.
    std::string offset_mask = name + ".conv_offset_mask";
    int32_t offset_mask_channels = deformable_groups * kernel.d[0] * kernel.d[1];

    nvinfer1::Weights offset_mask_weights = weight_map.at(offset_mask + ".weight");
    nvinfer1::Weights offset_mask_bias = weight_map.at(offset_mask + ".bias");

    offset_mask_weights.count = 2 * offset_mask_channels * channels * kernel.d[0] * kernel.d[1];
    offset_mask_bias.count = 2 * offset_mask_channels;

    auto conv_offset = network->addConvolutionNd(*feature,
                                                 offset_mask_channels * 2,
                                                 kernel,
                                                 offset_mask_weights,
                                                 offset_mask_bias);
    conv_offset->setStrideNd(stride);
    conv_offset->setPaddingNd(padding);
    setName(conv_offset, name + ".conv_offset");

    offset_mask_weights.values = static_cast<const float *>(offset_mask_weights.values) + offset_mask_weights.count;
    offset_mask_bias.values = static_cast<const float *>(offset_mask_bias.values) + offset_mask_bias.count;
    offset_mask_weights.count = offset_mask_channels * channels * kernel.d[0] * kernel.d[1];
    offset_mask_bias.count = offset_mask_channels;

    auto conv_mask = network->addConvolutionNd(*feature,
                                               offset_mask_channels,
                                               kernel,
                                               offset_mask_weights,
                                               offset_mask_bias);

    conv_mask->setStrideNd(stride);
    conv_mask->setPaddingNd(padding);
    setName(conv_offset, name + ".conv_mask");
    auto mask = network->addActivation(*conv_mask->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    setName(mask, name + ".conv_mask+activation(sigmoid)");

    const int32_t dilation[2] = {1, 1};
    const float beta = 0;
    nvinfer1::PluginField fs[]{
        {"stride", stride.d, nvinfer1::PluginFieldType::kINT32, 2},
        {"padding", padding.d, nvinfer1::PluginFieldType::kINT32, 2},
        {"dilation", dilation, nvinfer1::PluginFieldType::kINT32, 2},
        {"deformable_groups", &deformable_groups, nvinfer1::PluginFieldType::kINT32, 1},
        {"activation_type", &type, nvinfer1::PluginFieldType::kINT32, 1},
        {"alpha", &alpha, nvinfer1::PluginFieldType::kFLOAT32, 1},
        {"beta", &beta, nvinfer1::PluginFieldType::kFLOAT32, 1}};
    nvinfer1::PluginFieldCollection fc{7, fs};

    auto &dcn_weight = weight_map.at(name + ".weight");
    auto &dcn_bias = weight_map.at(name + ".bias");
    nvinfer1::ITensor *inputs[5]{
        input,
        conv_offset->getOutput(0),
        mask->getOutput(0),
        network->addConstant(nvinfer1::Dims{4,
                                            {channels,
                                             int32_t(dcn_weight.count / channels / kernel.d[0] / kernel.d[1]),
                                             kernel.d[0],
                                             kernel.d[1]}},
                             dcn_weight)
            ->getOutput(0),
        network->addConstant(nvinfer1::Dims{1, {int32_t(dcn_bias.count)}}, dcn_bias)->getOutput(0)};

    auto dcn = network->addPluginV2(inputs,
                                    5,
                                    *plugins->getPluginCreator("DCNLayer", "1", "")->createPlugin("DCNLayer", &fc));
    setName(dcn, name);
    return dcn->getOutput(0);
  }

  template<typename Kernel, typename Stride, typename Padding>
  nvinfer1::ITensor *makeDeformableConv2d(const std::string &name,
                                          nvinfer1::ITensor *input,
                                          nvinfer1::ITensor *feature,
                                          int32_t channels,
                                          Kernel k,
                                          Stride s,
                                          Padding p,
                                          int32_t deformable_groups) {
    return makeDeformableConv2dAc(name, input, feature, channels, k, s, p, deformable_groups, nvinfer1::ActivationType(-1), 0);
  }

  template<typename Kernel, typename Stride, typename Padding>
  nvinfer1::ITensor *makeDeformableConv2dLeakyReLU(const std::string &name,
                                                   nvinfer1::ITensor *input,
                                                   nvinfer1::ITensor *feature,
                                                   int32_t channels,
                                                   Kernel k,
                                                   Stride s,
                                                   Padding p,
                                                   int32_t deformable_groups,
                                                   float alpha) {
    return makeDeformableConv2dAc(name, input, feature, channels, k, s, p, deformable_groups, nvinfer1::ActivationType::kLEAKY_RELU, alpha);
  }

  nvinfer1::ITensor *makeLeakyReLU(nvinfer1::ITensor *input, float alpha, const std::string &name = "") {
    auto relu = network->addActivation(*input, nvinfer1::ActivationType::kLEAKY_RELU);
    relu->setAlpha(alpha);
    if (!name.empty()) {
      setName(relu, name);
    }
    return relu->getOutput(0);
  }

  template<typename Kernel, typename Stride, typename Padding>
  nvinfer1::ITensor *makeConv2dLeakyReLU(const std::string &name,
                                         nvinfer1::ITensor *input,
                                         int32_t channels,
                                         Kernel k,
                                         Stride s,
                                         Padding p,
                                         float alpha) {
    auto conv = makeConv2d(name, input, channels, k, s, p);
    return makeLeakyReLU(conv, alpha, name + "+activation(lrelu)");
  }

  [[nodiscard]] nvinfer1::ITensor *makeConcatenation(std::vector<nvinfer1::ITensor *> inputs, int32_t axis, const std::string &name = "") {
    assert(inputs.size() != 0);
    if (inputs.size() == 1) {
      return inputs[0];
    }
    auto concat = network->addConcatenation(inputs.data(), int32_t(inputs.size()));
    concat->setAxis(axis);
    if (!name.empty()) {
      setName(concat, name);
    }
    return concat->getOutput(0);
  }

  [[nodiscard]] nvinfer1::ITensor *makeReshape(nvinfer1::ITensor *input, nvinfer1::Dims dims, const std::string &name = "") {
    auto shuffle = network->addShuffle(*input);
    shuffle->setReshapeDimensions(dims);
    if (!name.empty()) {
      setName(shuffle, name);
    }
    return shuffle->getOutput(0);
  }

  [[nodiscard]] nvinfer1::ITensor *makeReshapeGather(nvinfer1::ITensor *input, const std::vector<int32_t> &idx, const std::string &name = "") {
    auto shape_layer = network->addShape(*input);
    if (!name.empty()) {
      setName(shape_layer, name + "+shape");
    }
    auto shape = shape_layer->getOutput(0);
    std::vector<nvinfer1::ITensor *> dims;
    for (auto ref = idx.begin(); ref != idx.end(); ++ref) {
      if (*ref == -1) {
        dims.push_back(network->addConstant({1, {1}}, minus1)->getOutput(0));
      }
      else {
        dims.push_back(network->addSlice(*shape, {1, {int32_t(ref - idx.begin())}}, {1, {1}}, {1, {1}})->getOutput(0));
      }
    }

    auto shuffle = network->addShuffle(*input);
    shuffle->setInput(1, *makeConcatenation(dims, 0, name.empty() ? name : name + "+shapeConcat"));
    if (!name.empty()) {
      setName(shuffle, name);
    }
    return shuffle->getOutput(0);
  }

  [[nodiscard]] nvinfer1::ITensor *makeStack(std::vector<nvinfer1::ITensor *> inputs, int32_t axis, const std::string &name = "") {
    for (auto &input: inputs) {
      auto dims = input->getDimensions();
      assert(axis < dims.nbDims);
      std::copy_backward(&dims.d[axis], &dims.d[dims.nbDims], &dims.d[axis + 1]);
      dims.d[axis] = 1;
      auto shuffle = network->addShuffle(*input);
      shuffle->setReshapeDimensions(dims);
      if (!name.empty()) {
        setName(shuffle, name + "+reshape#" + std::to_string(&input - inputs.data()));
      }
      input = shuffle->getOutput(0);
    }
    return makeConcatenation(inputs, axis, name);
  }

  nvinfer1::ITensor *makeBilinearResize(nvinfer1::ITensor *input, float scale, const std::string &name = "") {
    auto resize = network->addResize(*input);
    resize->setResizeMode(nvinfer1::ResizeMode::kLINEAR);
    resize->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL);
    resize->setSelectorForSinglePixel(nvinfer1::ResizeSelector::kUPPER);

    float scales[4] = {1, 1, scale, scale};
    resize->setScales(scales, 4);
    if (!name.empty()) {
      setName(resize, name);
    }
    return resize->getOutput(0);
  }

  nvinfer1::ITensor *makeElementWiseConstant(nvinfer1::ITensor *input, nvinfer1::ElementWiseOperation op, float constant, const std::string &name = "") {
    auto tmp = std::make_unique<float[]>(1);
    tmp[0] = constant;
    nvinfer1::Weights c{nvinfer1::DataType::kFLOAT, tmp.get(), 1};
    buffers.emplace_back(std::move(tmp));

    nvinfer1::Dims dims{input->getDimensions().nbDims, {}};
    for (int32_t i = 0; i < dims.nbDims; ++i) {
      dims.d[i] = 1;
    }

    auto ew = network->addElementWise(*input, *network->addConstant(dims, c)->getOutput(0), op);
    if (!name.empty()) {
      setName(ew, name);
    }
    return ew->getOutput(0);
  }

  nvinfer1::ITensor *makeElementWiseArray(nvinfer1::ITensor *input, nvinfer1::ElementWiseOperation op, const std::vector<float> &constant, const std::string &name = "") {
    auto tmp = std::make_unique<float[]>(constant.size());
    memcpy(tmp.get(), constant.data(), constant.size() * sizeof(float));
    nvinfer1::Weights c{nvinfer1::DataType::kFLOAT, tmp.get(), static_cast<int64_t>(constant.size())};
    buffers.emplace_back(std::move(tmp));

    nvinfer1::Dims dims{input->getDimensions().nbDims, {}};
    for (int32_t i = 0; i < dims.nbDims - 1; ++i) {
      dims.d[i] = 1;
    }
    dims.d[dims.nbDims - 1] = constant.size();

    auto ew = network->addElementWise(*input, *network->addConstant(dims, c)->getOutput(0), op);
    if (!name.empty()) {
      setName(ew, name);
    }
    return ew->getOutput(0);
  }

  nvinfer1::ITensor *repeat(nvinfer1::ITensor *input,
                            const std::function<nvinfer1::ITensor *(nvinfer1::ITensor *, uint32_t)> &block,
                            uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
      input = block(input, i);
    }

    return input;
  }

 private:
  static const int32_t _minus1 = -1;
  static constexpr nvinfer1::Weights minus1{nvinfer1::DataType::kINT32, &_minus1, 1};

  static nvinfer1::DimsHW asDims(nvinfer1::DimsHW d) {
    return d;
  }

  template<class I>
  static nvinfer1::DimsHW asDims(I d) {
    return nvinfer1::DimsHW{static_cast<int32_t>(d), static_cast<int32_t>(d)};
  }

  void setName(nvinfer1::ILayer *layer, const std::string &name) {
    if (name.empty()) {
      return;
    }

    auto ref = naming.find(name);
    if (ref == naming.end()) {
      layer->setName(name.c_str());
      naming.emplace(name, 1);
    }
    else {
      layer->setName((name + "$" + std::to_string(ref->second)).c_str());
      ++ref->second;
    }
  }
};

// nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kSTANDARD);
nvinfer1::IPluginRegistry *NetworkDefinitionHelper::plugins = getPluginRegistry();

static Logger gLogger;

struct OptimizationContext {
  OptimizationConfig config;

  nvinfer1::DataType ioDataType;

  nvinfer1::IBuilder *builder;
  WeightMap weights;
  nvinfer1::ITimingCache *cache;

  void init_cache() {
    auto conf = builder->createBuilderConfig();

    std::ifstream input("outputs/timing.cache", std::ios::binary | std::ios::in);
    if (input.is_open()) {
      auto size = std::filesystem::file_size("outputs/timing.cache");
      auto *values = new char[size];
      input.read(values, size);
      cache = conf->createTimingCache(values, size);
      delete[] values;
      input.close();
    }
    else {
      cache = conf->createTimingCache(nullptr, 0);
    }
  }

  void save_cache() const {
    if (cache == nullptr) {
      return;
    }

    std::ofstream output("outputs/timing.cache", std::ios::binary | std::ios::out);
    auto memory = cache->serialize();
    output.write(static_cast<char *>(memory->data()), memory->size());
    output.close();
  }

  nvinfer1::IBuilderConfig *prepareConfig() const {
    auto conf = builder->createBuilderConfig();
    if (config.use_fp16) { conf->setFlag(nvinfer1::BuilderFlag::kFP16); }
    conf->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
    conf->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    // /usr/src/tensorrt/bin/trtexec --verbose --noDataTransfers --useCudaGraph --separateProfileRun --useSpinWait --nvtxMode=verbose --loadEngine=./mutual_cycle.engine --exportTimes=./mutual_cycle.timing.json --exportProfile=./mutual_cycle.profile.json --exportLayerInfo=./mutual_cycle.graph.json --timingCacheFile=./timing.cache --best --avgRuns=1000 "--shapes=lf0:1x64x180x270,lf1:1x64x180x270,lf2:1x64x180x270"
    conf->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    conf->setTimingCache(*cache, false);

    return conf;
  }

  nvinfer1::INetworkDefinition *createNetwork() const {
    return builder->createNetworkV2(1u << uint32_t(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  }
};

const char *InputFeatureExtract = "frame";
const char *OutputFeatureExtract[] = {"l1", "l2", "l3"};

void buildFeatureExtract(OptimizationContext &ctx, const std::string &model_path, const std::string &save_path) {
  auto network = ctx.createNetwork();

  //  {
  //    auto input = network->addInput(InputFeatureExtract, ioDataType, nvinfer1::Dims4{-1, -1, -1, 3});
  //
  //    input = helper.makeElementWiseArray(input, nvinfer1::ElementWiseOperation::kSUB, {0.485, 0.456, 0.406}, "normalize-mean");
  //    input = helper.makeElementWiseArray(input, nvinfer1::ElementWiseOperation::kDIV, {0.229, 0.224, 0.225}, "normalize-std");
  //
  //    // interleave to planar
  //    auto shuffle_in = network->addShuffle(*input);
  //    shuffle_in->setName("uninterleave");
  //    shuffle_in->setFirstTranspose({0, 3, 1, 2});
//    auto tensor = shuffle_in->getOutput(0);
//
//    // L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
//    tensor = helper.makeConv2dLeakyReLU("conv_first", tensor, feature_count, 3, 1, 1, 0.1);
//
//    // L1_fea = self.feature_extraction(L1_fea) (ResidualBlock_noBN * 5)
//    for (uint32_t i = 0; i < 5; ++i) {
//      auto name = "feature_extraction." + std::to_string(i);
//      auto original = tensor;
//      tensor = helper.makeConv2d(name + ".conv1", tensor, feature_count, 3, 1, 1);
//      tensor = network->addActivation(*tensor, nvinfer1::ActivationType::kRELU)->getOutput(0);
//      tensor = helper.makeConv2d(name + ".conv2", tensor, feature_count, 3, 1, 1);
//      tensor = network->addElementWise(*tensor, *original, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
//    }
//
//    auto l1 = tensor;
//
//    l1->setName(OutputFeatureExtract[0]);
//    network->markOutput(*l1);
//    l1->setType(ioDataType);
//
//    // L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
//    // L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
//
//    auto l2 = helper.makeConv2dLeakyReLU("fea_L2_conv1", l1, feature_count, 3, 2, 1, 0.1);
//    l2 = helper.makeConv2dLeakyReLU("fea_L2_conv2", l2, feature_count, 3, 1, 1, 0.1);
//
//    l2->setName(OutputFeatureExtract[1]);
//    network->markOutput(*l2);
//    l2->setType(ioDataType);
//
//    // L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
//    // L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))
//
//    auto l3 = helper.makeConv2dLeakyReLU("fea_L3_conv1", l2, feature_count, 3, 2, 1, 0.1);
//    l3 = helper.makeConv2dLeakyReLU("fea_L3_conv2", l3, feature_count, 3, 1, 1, 0.1);
//
//    l3->setName(OutputFeatureExtract[2]);
//    network->markOutput(*l3);
//    l3->setType(ioDataType);
//  }

  {
    auto parser = nvonnxparser::createParser(*network, gLogger);
    //    parser->parseFromFile("model_src/cycmunet-fe-dy-processed.onnx", 1);
    parser->parseFromFile(model_path.c_str(), 1);

    network->getInput(0)->setName(InputFeatureExtract);
    network->getOutput(0)->setName(OutputFeatureExtract[0]);
    network->getOutput(1)->setName(OutputFeatureExtract[1]);
    network->getOutput(1)->setName(OutputFeatureExtract[2]);

    network->getInput(0)->setType(ctx.ioDataType);
    network->getOutput(0)->setType(ctx.ioDataType);
    network->getOutput(1)->setType(ctx.ioDataType);
    network->getOutput(2)->setType(ctx.ioDataType);
  }

  std::cout << "Done define feature extract net." << std::endl;

  auto config = ctx.prepareConfig();
  //  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 512llu * 1024 * 1024);

  auto profile = ctx.builder->createOptimizationProfile();
  profile->setDimensions(
      InputFeatureExtract, nvinfer1::OptProfileSelector::kMIN,
      nvinfer1::Dims4{ctx.config.batch_extract.min, ctx.config.input_height.min, ctx.config.input_width.min, 3});
  profile->setDimensions(
      InputFeatureExtract, nvinfer1::OptProfileSelector::kOPT,
      nvinfer1::Dims4{ctx.config.batch_extract.opt, ctx.config.input_height.opt, ctx.config.input_width.opt, 3});
  profile->setDimensions(
      InputFeatureExtract, nvinfer1::OptProfileSelector::kMAX,
      nvinfer1::Dims4{ctx.config.batch_extract.max, ctx.config.input_height.max, ctx.config.input_width.max, 3});
  config->addOptimizationProfile(profile);

  auto modelStream = ctx.builder->buildSerializedNetwork(*network, *config);

  std::cout << "Done build feature extract net." << std::endl;

  //  std::ofstream p("models/feature_extract.engine", std::ios::binary);
  std::ofstream p(save_path, std::ios::binary);
  COND_CHECK(p.is_open(), "Unable to open engine file.");
  p.write(static_cast<const char *>(modelStream->data()), modelStream->size());
  p.close();
  std::cout << "Done save feature extract net." << std::endl;
}

const char *InputFeatureFusion[] = {"f0l1", "f0l2", "f0l3", "f2l1", "f2l2", "f2l3"};
const char *OutputFeatureFusion = "f1l1";

void buildFeatureFusion(OptimizationContext &ctx, const std::string &save_path) {
  auto network = ctx.createNetwork();
  NetworkDefinitionHelper helper{network, ctx.weights};
  auto feature_count = ctx.config.feature_count;
  auto deformable_groups = ctx.config.deformable_groups;

  {
    nvinfer1::ITensor *f1[] = {
        network->addInput(InputFeatureFusion[0], ctx.ioDataType, nvinfer1::Dims4{-1, feature_count, -1, -1}),
        network->addInput(InputFeatureFusion[1], ctx.ioDataType, nvinfer1::Dims4{-1, feature_count, -1, -1}),
        network->addInput(InputFeatureFusion[2], ctx.ioDataType, nvinfer1::Dims4{-1, feature_count, -1, -1})};

    nvinfer1::ITensor *f2[] = {
        network->addInput(InputFeatureFusion[3], ctx.ioDataType, nvinfer1::Dims4{-1, feature_count, -1, -1}),
        network->addInput(InputFeatureFusion[4], ctx.ioDataType, nvinfer1::Dims4{-1, feature_count, -1, -1}),
        network->addInput(InputFeatureFusion[5], ctx.ioDataType, nvinfer1::Dims4{-1, feature_count, -1, -1})};

    auto makePCDAlign = [&](nvinfer1::ITensor *f1[3], nvinfer1::ITensor *f2[3], const std::string &suffix) -> nvinfer1::ITensor * {
      // clang-format off

      // L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
      // L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
      // L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset))
      // L3_fea = self.lrelu(self.L3_dcnpack_1(fea1[2], L3_offset))
      auto l3_offset = helper.makeConcatenation({f1[2], f2[2]}, 1);
           l3_offset = helper.makeConv2dLeakyReLU("pcd_align.L3_offset_conv1" + suffix, l3_offset, feature_count, 3, 1, 1, 0.1);
           l3_offset = helper.makeConv2dLeakyReLU("pcd_align.L3_offset_conv2" + suffix, l3_offset, feature_count, 3, 1, 1, 0.1);
      auto l3_fea = helper.makeDeformableConv2dLeakyReLU("pcd_align.L3_dcnpack" + suffix, f1[2], l3_offset, feature_count, 3, 1, 1, deformable_groups, 0.1);

      //  L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
      //  L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
      //  L3_offset = F.interpolate(
      //      L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
      //  L2_offset = self.lrelu(self.L2_offset_conv2_1(
      //      torch.cat([L2_offset, L3_offset * 2], dim=1)))
      //  L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset))
      //  L2_fea = self.L2_dcnpack_1(fea1[1], L2_offset)
      //  L3_fea = F.interpolate(L3_fea, scale_factor=2,
      //                         mode='bilinear', align_corners=False)
      //  L2_fea = self.lrelu(self.L2_fea_conv_1(
      //      torch.cat([L2_fea, L3_fea], dim=1)))
      auto l2_offset = helper.makeConcatenation({f1[1], f2[1]}, 1);
           l2_offset = helper.makeConv2dLeakyReLU("pcd_align.L2_offset_conv1" + suffix, l2_offset, feature_count, 3, 1, 1, 0.1);
           l3_offset = helper.makeBilinearResize(l3_offset, 2);
           l3_offset = helper.makeElementWiseConstant(l3_offset, nvinfer1::ElementWiseOperation::kPROD, 2);
           l2_offset = helper.makeConcatenation({l2_offset, l3_offset}, 1);
           l2_offset = helper.makeConv2dLeakyReLU("pcd_align.L2_offset_conv2" + suffix, l2_offset, feature_count, 3, 1, 1, 0.1);
           l2_offset = helper.makeConv2dLeakyReLU("pcd_align.L2_offset_conv3" + suffix, l2_offset, feature_count, 3, 1, 1, 0.1);
      auto l2_fea = helper.makeDeformableConv2d("pcd_align.L2_dcnpack" + suffix, f1[1], l2_offset, feature_count, 3, 1, 1, deformable_groups);
           l3_fea = helper.makeBilinearResize(l3_fea, 2);
           l2_fea = helper.makeConcatenation({l2_fea, l3_fea}, 1);
           l2_fea = helper.makeConv2dLeakyReLU("pcd_align.L2_fea_conv" + suffix, l2_fea, feature_count, 3, 1, 1, 0.1);

      auto l1_offset = helper.makeConcatenation({f1[0], f2[0]}, 1);
           l1_offset = helper.makeConv2dLeakyReLU("pcd_align.L1_offset_conv1" + suffix, l1_offset, feature_count, 3, 1, 1, 0.1);
           l2_offset = helper.makeBilinearResize(l2_offset, 2);
           l2_offset = helper.makeElementWiseConstant(l2_offset, nvinfer1::ElementWiseOperation::kPROD, 2);
           l1_offset = helper.makeConcatenation({l1_offset, l2_offset}, 1);
           l1_offset = helper.makeConv2dLeakyReLU("pcd_align.L1_offset_conv2" + suffix, l1_offset, feature_count, 3, 1, 1, 0.1);
           l1_offset = helper.makeConv2dLeakyReLU("pcd_align.L1_offset_conv3" + suffix, l1_offset, feature_count, 3, 1, 1, 0.1);
      auto l1_fea = helper.makeDeformableConv2d("pcd_align.L1_dcnpack" + suffix, f1[0], l1_offset, feature_count, 3, 1, 1, deformable_groups);
           l2_fea = helper.makeBilinearResize(l2_fea, 2);
           l1_fea = helper.makeConcatenation({l1_fea, l2_fea}, 1);
           l1_fea = helper.makeConv2d("pcd_align.L1_fea_conv" + suffix, l1_fea, feature_count, 3, 1, 1);
      // clang-format on
      return l1_fea;
    };

    // aligned_fea = self.pcd_align(fea1, fea2)
    auto feature = helper.makeConcatenation({makePCDAlign(f1, f2, "_1"), makePCDAlign(f2, f1, "_2")}, 1);

    // fusion_fea = self.fusion(aligned_fea)
    feature = helper.makeConv2d("fusion", feature, feature_count, 1, 1, 0);
    feature->setName(OutputFeatureFusion);
    network->markOutput(*feature);
    feature->setType(ctx.ioDataType);
  }

  std::cout << "Done define feature fusion net." << std::endl;

  auto config = ctx.prepareConfig();
  //  config->setMaxWorkspaceSize(4096llu * 1024 * 1024);
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 8192llu * 1024 * 1024);

  // clang-format off
  auto profile = ctx.builder->createOptimizationProfile();
  profile->setDimensions(InputFeatureFusion[0], nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ctx.config.batch_fusion.min, feature_count, ctx.config.input_height.min, ctx.config.input_width.min});
  profile->setDimensions(InputFeatureFusion[0], nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ctx.config.batch_fusion.opt, feature_count, ctx.config.input_height.opt, ctx.config.input_width.opt});
  profile->setDimensions(InputFeatureFusion[0], nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ctx.config.batch_fusion.max, feature_count, ctx.config.input_height.max, ctx.config.input_width.max});
  profile->setDimensions(InputFeatureFusion[1], nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ctx.config.batch_fusion.min, feature_count, ctx.config.input_height.min / 2, ctx.config.input_width.min / 2});
  profile->setDimensions(InputFeatureFusion[1], nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ctx.config.batch_fusion.opt, feature_count, ctx.config.input_height.opt / 2, ctx.config.input_width.opt / 2});
  profile->setDimensions(InputFeatureFusion[1], nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ctx.config.batch_fusion.max, feature_count, ctx.config.input_height.max / 2, ctx.config.input_width.max / 2});
  profile->setDimensions(InputFeatureFusion[2], nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ctx.config.batch_fusion.min, feature_count, ctx.config.input_height.min / 4, ctx.config.input_width.min / 4});
  profile->setDimensions(InputFeatureFusion[2], nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ctx.config.batch_fusion.opt, feature_count, ctx.config.input_height.opt / 4, ctx.config.input_width.opt / 4});
  profile->setDimensions(InputFeatureFusion[2], nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ctx.config.batch_fusion.max, feature_count, ctx.config.input_height.max / 4, ctx.config.input_width.max / 4});
  profile->setDimensions(InputFeatureFusion[3], nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ctx.config.batch_fusion.min, feature_count, ctx.config.input_height.min, ctx.config.input_width.min});
  profile->setDimensions(InputFeatureFusion[3], nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ctx.config.batch_fusion.opt, feature_count, ctx.config.input_height.opt, ctx.config.input_width.opt});
  profile->setDimensions(InputFeatureFusion[3], nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ctx.config.batch_fusion.max, feature_count, ctx.config.input_height.max, ctx.config.input_width.max});
  profile->setDimensions(InputFeatureFusion[4], nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ctx.config.batch_fusion.min, feature_count, ctx.config.input_height.min / 2, ctx.config.input_width.min / 2});
  profile->setDimensions(InputFeatureFusion[4], nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ctx.config.batch_fusion.opt, feature_count, ctx.config.input_height.opt / 2, ctx.config.input_width.opt / 2});
  profile->setDimensions(InputFeatureFusion[4], nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ctx.config.batch_fusion.max, feature_count, ctx.config.input_height.max / 2, ctx.config.input_width.max / 2});
  profile->setDimensions(InputFeatureFusion[5], nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ctx.config.batch_fusion.min, feature_count, ctx.config.input_height.min / 4, ctx.config.input_width.min / 4});
  profile->setDimensions(InputFeatureFusion[5], nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ctx.config.batch_fusion.opt, feature_count, ctx.config.input_height.opt / 4, ctx.config.input_width.opt / 4});
  profile->setDimensions(InputFeatureFusion[5], nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ctx.config.batch_fusion.max, feature_count, ctx.config.input_height.max / 4, ctx.config.input_width.max / 4});
  config->addOptimizationProfile(profile);
  // clang-format on

  auto modelStream = ctx.builder->buildSerializedNetwork(*network, *config);

  std::cout << "Done build feature fusion net." << std::endl;

//  std::ofstream p("models/feature_fusion.engine", std::ios::binary);
  std::ofstream p(save_path, std::ios::binary);
  COND_CHECK(p.is_open(), "Unable to open engine file.");
  p.write(static_cast<const char *>(modelStream->data()), modelStream->size());
  p.close();
  std::cout << "Done save feature fusion net." << std::endl;
}

const char *InputMutualCycle[] = {"lf0", "lf1", "lf2"};
const char *OutputMutualCycle[] = {"h0", "h1", "h2"};

void buildMutualCycle(OptimizationContext &ctx, const std::string &model_path, const std::string &save_path) {
  //  auto network = builder->createNetworkV2(1u << uint32_t(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  //  NetworkDefinitionHelper helper{network, weights};
  //
  //  auto makeProAlign = [&](const std::string &name, nvinfer1::ITensor *input) {
  //    auto l_prev2 = helper.makeConv2dLeakyReLU(name + ".conv3x3", input, feature_count, 3, 1, 1, 0.1);
  //    auto l_prev3 = helper.makeReshapeGather(l_prev2, {0, -1, 3, 4});
  //    l_prev3 = helper.makeConv2dLeakyReLU(name + ".conv1x1", input, feature_count, 1, 1, 1, 0.1);
  //
  //  };
  //
  //  auto makeSR = [&](const std::string &name, nvinfer1::ITensor *input) {
  //    makeProAlign(name, input);
  //  };
  //
  //  std::vector<nvinfer1::ITensor *> l1_out{
  //      network->addInput(InputMutualCycle[0], nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{-1, feature_count, -1, -1})};
  //  std::vector<nvinfer1::ITensor *> l2_out{
  //      network->addInput(InputMutualCycle[1], nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{-1, feature_count, -1, -1})};
  //  std::vector<nvinfer1::ITensor *> l3_out{
  //      network->addInput(InputMutualCycle[2], nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{-1, feature_count, -1, -1})};
  //
  //  auto l_feats = helper.makeStack({
  //                                      helper.makeConcatenation(l1_out, 1),
  //                                      helper.makeConcatenation(l2_out, 1),
  //                                      helper.makeConcatenation(l3_out, 1),
  //                                  },
  //                                  0);
  //  l_feats = helper.makeConv2dLeakyReLU("merge.0", l_feats, feature_count, 1, 1, 1, 0.1);

  auto network = ctx.createNetwork();
  {
    auto parser = nvonnxparser::createParser(*network, gLogger);
//    parser->parseFromFile("model_src/cycmunet-mu-dy-processed.onnx", 1);
    parser->parseFromFile(model_path.c_str(), 1);

    network->getInput(0)->setName(InputMutualCycle[0]);
    network->getInput(1)->setName(InputMutualCycle[1]);
    network->getInput(2)->setName(InputMutualCycle[2]);
    network->getOutput(0)->setName(OutputMutualCycle[0]);
    network->getOutput(1)->setName(OutputMutualCycle[1]);

    network->getInput(0)->setType(ctx.ioDataType);
    network->getInput(1)->setType(ctx.ioDataType);
    network->getInput(2)->setType(ctx.ioDataType);
    network->getOutput(0)->setType(ctx.ioDataType);
    network->getOutput(1)->setType(ctx.ioDataType);
  }

  std::cout << "Done define mutual cycle net." << std::endl;
  auto config = ctx.prepareConfig();
  // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 12llu * 1024 * 1024 * 1024);

  // clang-format off
  auto profile = ctx.builder->createOptimizationProfile();
  profile->setDimensions(InputMutualCycle[0], nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ctx.config.batch_cycle.min, ctx.config.feature_count, ctx.config.input_height.min, ctx.config.input_width.min});
  profile->setDimensions(InputMutualCycle[0], nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ctx.config.batch_cycle.opt, ctx.config.feature_count, ctx.config.input_height.opt, ctx.config.input_width.opt});
  profile->setDimensions(InputMutualCycle[0], nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ctx.config.batch_cycle.max, ctx.config.feature_count, ctx.config.input_height.max, ctx.config.input_width.max});
  profile->setDimensions(InputMutualCycle[1], nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ctx.config.batch_cycle.min, ctx.config.feature_count, ctx.config.input_height.min, ctx.config.input_width.min});
  profile->setDimensions(InputMutualCycle[1], nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ctx.config.batch_cycle.opt, ctx.config.feature_count, ctx.config.input_height.opt, ctx.config.input_width.opt});
  profile->setDimensions(InputMutualCycle[1], nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ctx.config.batch_cycle.max, ctx.config.feature_count, ctx.config.input_height.max, ctx.config.input_width.max});
  profile->setDimensions(InputMutualCycle[2], nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ctx.config.batch_cycle.min, ctx.config.feature_count, ctx.config.input_height.min, ctx.config.input_width.min});
  profile->setDimensions(InputMutualCycle[2], nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ctx.config.batch_cycle.opt, ctx.config.feature_count, ctx.config.input_height.opt, ctx.config.input_width.opt});
  profile->setDimensions(InputMutualCycle[2], nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ctx.config.batch_cycle.max, ctx.config.feature_count, ctx.config.input_height.max, ctx.config.input_width.max});
  config->addOptimizationProfile(profile);
// clang-format on

  auto modelStream = ctx.builder->buildSerializedNetwork(*network, *config);
  COND_CHECK(modelStream->data() != nullptr, "No model build output.");

  std::cout << "Done build mutual cycle net." << std::endl;

//  std::ofstream p("models/mutual_cycle.engine", std::ios::binary);
  std::ofstream p(save_path, std::ios::binary);
  COND_CHECK(p.is_open(), "Unable to open engine file.");
  p.write(static_cast<const char *>(modelStream->data()), modelStream->size());
  p.close();
  std::cout << "Done save mutual cycle net." << std::endl;
}

// feature extract:
//   0   2   4   6
//         |
//   0   2   4   6 (3x)

// feature fusion
//   0   2   4 (3x)
//   2   4   6 (3x)
//       |
//   1   3   5

// main
//   0   2   4
//   1   3   5
//   2   4   6
//       |
// HR of input

bool optimize(OptimizationConfig config) {
  UDOLayers::registerPlugins();

  OptimizationContext ctx {};
  ctx.config = config;
  ctx.ioDataType = ctx.config.use_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;

  ctx.builder = nvinfer1::createInferBuilder(gLogger);
  ctx.weights = loadWeights("model_src/weights.pb");
  std::cout << "Weights loaded." << std::endl;

  ctx.init_cache();

  buildFeatureExtract(ctx, "model_src/cycmunet-fe-dy-processed.onnx", "models/feature_extract.engine");
  buildFeatureFusion(ctx, "models/feature_fusion.engine");
  buildMutualCycle(ctx, "model_src/cycmunet-mu-dy-processed.onnx", "models/mutual_cycle.engine");

  ctx.save_cache();

  return true;
}

int main() {
  OptimizationConfig config {
        320, 180,
        2, 1, 1,
        64, 8,
        false
  };

  COND_CHECK(optimize(config), "Build failed");
}
