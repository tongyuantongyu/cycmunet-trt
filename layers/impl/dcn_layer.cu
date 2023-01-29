#include <iostream>

#include "dcn_layer_impl.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cassert>
//#include "reveal.h"

#ifdef __LP64__
template<>
template<>
util_attrs hw<half>::operator hw<offset_t>() const noexcept {
  return {static_cast<offset_t>(static_cast<long long>(h)),
          static_cast<offset_t>(static_cast<long long>(w))};
}
#endif

constexpr std::size_t threadCount = 1024;
constexpr std::size_t threadCountIm2Col = 512;

struct im2col_parameters {
  hw<> stride;
  hw<> padding;
  hw<> dilation;
  int32_t channel_per_deformable_group;
};

template<class F>
struct bias_activation_parameters {
  // see NvInfer.h ActivationType
  int32_t activation_type;
  F alpha, beta;
};

half __device__ floor(const half f) {
  return hfloor(f);
}

half __device__ ceil(const half f) {
  return hceil(f);
}

// Gather data from coordination. Use bilinear interpolation.
template<class F>
static inline F __device__ DCNGather(const md_view<const F, 4> &input,
                                      offset_t n,
                                      offset_t c,
                                      hw<offset_t> pos,
                                      hw<F> offset) {
  const F z{};
  F result{};

  const auto [h, w] = input.shape.template slice<2, 2>();
  const hw<F> ol_float{floor(offset.h), floor(offset.w)};
  const hw<F> oh_float{ceil(offset.h), ceil(offset.w)};

  const hw<offset_t> pl = pos + hw<offset_t>(ol_float), ph = pos + hw<offset_t>(oh_float);
  if (ph.h < 0 || ph.w < 0 || pl.h >= h || pl.w >= w) {
    return result;
  }

  // w(eight) of data at l(ow)/h(igh) pos
  const hw<F> wh = offset - ol_float;
  const hw<F> wl = -wh + 1;

  // should we read data at l(ow)/h(igh) h(eight)/w(idth) pos
  const bool lh = wl.h != z && pl.h >= 0;
  const bool lw = wl.w != z && pl.w >= 0;
  const bool hh = wh.h != z && ph.h < h;
  const bool hw = wh.w != z && ph.w < w;
  if (lh && lw) {
    result += input.at(n, c, pl.h, pl.w) * wl.h * wl.w;
  }
  if (lh && hw) {
    result += input.at(n, c, pl.h, ph.w) * wl.h * wh.w;
  }
  if (hh && lw) {
    result += input.at(n, c, ph.h, pl.w) * wh.h * wl.w;
  }
  if (hh && hw) {
    result += input.at(n, c, ph.h, ph.w) * wh.h * wh.w;
  }

  return result;
}

// Gather data from input into matrix form.
template<class F, uint32_t K>
static void __global__ DCNIm2colKernel(md_view<const F, 4> input, md_view<const F, 7> offset, md_view<const F, 6> mask,
                                       md_view<F, 6> col, im2col_parameters p, offset_t count) {
  offset_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= count) { return; }

  const auto [n, c, h, w] = col.shape.template gather<0, 1, 4, 5>().indexes(idx);
  // kernel h, w
  const auto [kh, kw] = col.shape.template slice<2, 2>();
  // index of deformable group
  const auto g = c / p.channel_per_deformable_group;
  // input h, w base offset
  const auto hin = h * p.stride.h - p.padding.h;
  const auto win = w * p.stride.w - p.padding.w;

#pragma unroll K
  for (uint32_t i = 0; i < uint32_t(kh); ++i) {
#pragma unroll K
    for (uint32_t j = 0; j < uint32_t(kw); ++j) {
      F data = DCNGather(input, n, c, {hin + i * p.dilation.h, win + j * p.dilation.w},
                         {offset.at(n, g, i, j, 0, h, w), offset.at(n, g, i, j, 1, h, w)});
      col.at(n, c, i, j, h, w) = data * mask.at(n, g, i, j, h, w);
    }
  }
}

// Broadcast bias to output result.
template<class F>
static void __global__ BiasBroadcastKernel(md_view<F, 4> output,
                                           md_view<const F, 1> bias,
                                           offset_t count) {
  offset_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= count) {
    return;
  }

  const auto [n, c, h, w] = output.shape.indexes(idx);
  output.at(n, c, h, w) = bias.at(c);
}

// Add bias to output result.
template<class F>
static void __global__ BiasActivationKernel(md_view<F, 4> output,
                                            md_view<const F, 1> bias,
                                            bias_activation_parameters<F> p,
                                            offset_t count) {
  offset_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= count) {
    return;
  }

  const auto [n, c, h, w] = output.shape.indexes(idx);
  F v = output.at(n, c, h, w) + bias.at(c);

  // support other when there's need
  assert(p.activation_type == 3);

  // leaky_relu
  if (v < F{}) {
    v *= p.alpha;
  }

  output.at(n, c, h, w) = v;
}


static cublasStatus_t cublasGemm(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const float *alpha, /* host or device pointer */
                                 const float *A,
                                 int lda,
                                 const float *B,
                                 int ldb,
                                 const float *beta, /* host or device pointer */
                                 float *C,
                                 int ldc) {
  return cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

static cublasStatus_t cublasGemm(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const half *alpha, /* host or device pointer */
                                 const half *A,
                                 int lda,
                                 const half *B,
                                 int ldb,
                                 const half *beta, /* host or device pointer */
                                 half *C,
                                 int ldc) {
  return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

static cublasStatus_t cublasGemmSB(cublasHandle_t handle,
                                   cublasOperation_t transa,
                                   cublasOperation_t transb,
                                   int m,
                                   int n,
                                   int k,
                                   const float *alpha, /* host or device pointer */
                                   const float *A,
                                   int lda,
                                   long long int strideA,
                                   const float *B,
                                   int ldb,
                                   long long int strideB,
                                   const float *beta, /* host or device pointer */
                                   float *C,
                                   int ldc,
                                   long long int strideC,
                                   int batchCount) {
  return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

static cublasStatus_t cublasGemmSB(cublasHandle_t handle,
                                   cublasOperation_t transa,
                                   cublasOperation_t transb,
                                   int m,
                                   int n,
                                   int k,
                                   const half *alpha, /* host or device pointer */
                                   const half *A,
                                   int lda,
                                   long long int strideA,
                                   const half *B,
                                   int ldb,
                                   long long int strideB,
                                   const half *beta, /* host or device pointer */
                                   half *C,
                                   int ldc,
                                   long long int strideC,
                                   int batchCount) {
  return cublasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

template<class F>
void compute(DCNLayerInput<F> inputs,
             DCNLayerOutput<F> outputs,
             DCNLayerConfig config,
             DCNLayerExtra extra,
             cudaStream_t stream) {
  assert(cudaGetLastError() == cudaSuccess);
  std::size_t count = inputs.im2col_buffer.shape.template gather<0, 1, 4, 5>().count();
  auto blocks = (count + threadCountIm2Col - 1) / threadCountIm2Col;

  im2col_parameters im2col_p{
      config.stride, config.padding, config.dilation,
      int32_t((inputs.input.shape[1] + config.deformable_groups - 1) / config.deformable_groups)};

  if (inputs.weight.shape[2] == 3 && inputs.weight.shape[3] == 3) {
    DCNIm2colKernel<F, 3><<<blocks, threadCountIm2Col, 0, stream>>>(inputs.input, inputs.offset, inputs.mask,
                                                                    inputs.im2col_buffer, im2col_p, count);
  }
  else if (inputs.weight.shape[2] == 1 && inputs.weight.shape[3] == 1) {
    DCNIm2colKernel<F, 1><<<blocks, threadCountIm2Col, 0, stream>>>(inputs.input, inputs.offset, inputs.mask,
                                                                    inputs.im2col_buffer, im2col_p, count);
  }
  else {
    DCNIm2colKernel<F, 2><<<blocks, threadCountIm2Col, 0, stream>>>(inputs.input, inputs.offset, inputs.mask,
                                                                    inputs.im2col_buffer, im2col_p, count);
  }

  assert(cudaGetLastError() == cudaSuccess);

  const int m = outputs.output.shape.template slice<2, 2>().count();
  const int n = outputs.output.shape[1];
  const int k = inputs.im2col_buffer.shape.template slice<1, 3>().count();

  F alpha = 1;
  F beta = 0;

  count = outputs.output.size();
  blocks = (count + threadCount - 1) / threadCount;

  // If activation not needed, broadcast bias and let gemm do the add for us.
  if (config.activation_type == -1) {
    BiasBroadcastKernel<<<blocks, threadCount, 0, stream>>>(outputs.output, inputs.bias, count);
    beta = 1;
    assert(cudaGetLastError() == cudaSuccess);
  }

  if (extra.blasMode == 0) {
    const auto cublasResult = cublasGemmSB(static_cast<cublasHandle_t>(extra.cublasHandle),
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_N,
                                           m,
                                           n,
                                           k,
                                           &alpha,
                                           inputs.im2col_buffer.data,
                                           m,
                                           inputs.im2col_buffer.shape.template slice<1, 5>().count(),
                                           inputs.weight.data,
                                           k,
                                           0,
                                           &beta,
                                           outputs.output.data,
                                           m,
                                           outputs.output.shape.template slice<1, 3>().count(),
                                           outputs.output.shape[0]);

    assert(cublasResult == CUBLAS_STATUS_SUCCESS);
  } else {
    for (offset_t i = 0; i < outputs.output.shape[0]; ++i) {
      const auto cublasResult = cublasGemm(static_cast<cublasHandle_t>(extra.cublasHandle),
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_N,
                                           m,
                                           n,
                                           k,
                                           &alpha,
                                           inputs.im2col_buffer.at(i).data,
                                           m,
                                           inputs.weight.data,
                                           k,
                                           &beta,
                                           outputs.output.at(i).data,
                                           m);

      assert(cublasResult == CUBLAS_STATUS_SUCCESS);
    }
  }

  // Fuse bias and activation, if there are.
  if (config.activation_type != -1) {
    bias_activation_parameters<F> ba_p{config.activation_type, F{config.alpha}, F{config.beta}};
    BiasActivationKernel<<<blocks, threadCount, 0, stream>>>(outputs.output, inputs.bias, ba_p, count);

    assert(cudaGetLastError() == cudaSuccess);
  }
}

// Explicit template instantiation. Keep these after template definition.
template void compute<float>(DCNLayerInput<float> inputs,
                             DCNLayerOutput<float> outputs,
                             DCNLayerConfig config,
                             DCNLayerExtra extra,
                             cudaStream_t stream);
template void compute<half>(DCNLayerInput<half> inputs,
                            DCNLayerOutput<half> outputs,
                            DCNLayerConfig config,
                            DCNLayerExtra extra,
                            cudaStream_t stream);