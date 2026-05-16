#include "blocks.h"
#include "isp_block.h"
#include <cmath>
#include <cstdio>
#include <memory>

// ============================================================================
// Gamma Correction
//
// Applies sRGB gamma curve to each channel of a float RGB image.
// Input:  RGB_FLOAT [0.0, 1.0]
// Output: RGB_FLOAT [0.0, 1.0] (in-place)
//
// sRGB formula:
//   if (x <= 0.0031308) out = 12.92 * x
//   else                out = 1.055 * pow(x, 1/2.4) - 0.055
// ============================================================================

__device__ float srgb_gamma(float x) {
  if (x <= 0.0031308f) {
    return 12.92f * x;
  } else {
    return 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
  }
}

__global__ void gamma_kernel(float *data, int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  float val = data[idx];
  // Clamp input to [0, 1] before gamma
  val = fminf(fmaxf(val, 0.0f), 1.0f);
  data[idx] = srgb_gamma(val);
}

// --- ISPBlock Implementation ---
class GammaCorrection : public ISPBlock {
public:
  GammaCorrection() = default;

  const char *name() const override { return "GammaCorrection (sRGB)"; }

  void process(const FrameBuffer &input, FrameBuffer &output,
               cudaStream_t stream) override {
    if (input.format != PixelFormat::RGB_FLOAT) {
      fprintf(stderr, "[Gamma] Error: expected RGB_FLOAT input\n");
      return;
    }

    // In-place operation
    output = input;

    int total = input.width * input.height * input.channels;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    gamma_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<float *>(output.d_data), total);
  }
};

// Factory function (called from main.cpp)
std::unique_ptr<ISPBlock> createGammaCorrection() {
  return std::make_unique<GammaCorrection>();
}

// ============================================================================
// Optimized Gamma Correction
// ============================================================================
__device__ float srgb_gamma_optimized(float x) {
  if (x <= 0.0031308f) {
    return 12.92f * x;
  } else {
    // 1.0f / 2.4f = 0.41666666f
    // Using __powf for fast hardware intrinsic math
    return 1.055f * __powf(x, 0.41666666f) - 0.055f;
  }
}

__global__ void gamma_kernel_float4(float4 *data, int total_f4_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_f4_elements)
    return;

  // Vectorized load (128-bit)
  float4 val = data[idx];

  // Clamp and apply gamma to each component
  val.x = srgb_gamma_optimized(fminf(fmaxf(val.x, 0.0f), 1.0f));
  val.y = srgb_gamma_optimized(fminf(fmaxf(val.y, 0.0f), 1.0f));
  val.z = srgb_gamma_optimized(fminf(fmaxf(val.z, 0.0f), 1.0f));
  val.w = srgb_gamma_optimized(fminf(fmaxf(val.w, 0.0f), 1.0f));

  // Vectorized store (128-bit)
  data[idx] = val;
}

class GammaCorrectionOptimized : public ISPBlock {
public:
  GammaCorrectionOptimized() = default;

  const char *name() const override { return "GammaCorrection (sRGB) - Optimized"; }

  void process(const FrameBuffer &input, FrameBuffer &output,
               cudaStream_t stream) override {
    if (input.format != PixelFormat::RGB_FLOAT) {
      fprintf(stderr, "[Gamma] Error: expected RGB_FLOAT input\n");
      return;
    }

    output = input;
    int total = input.width * input.height * input.channels;

    if (total % 4 != 0) {
      // Fallback to scalar kernel if total elements is not a multiple of 4
      int threads = 256;
      int blocks = (total + threads - 1) / threads;
      gamma_kernel<<<blocks, threads, 0, stream>>>(
          static_cast<float *>(output.d_data), total);
      return;
    }

    int total_f4 = total / 4;
    int threads = 256;
    int blocks = (total_f4 + threads - 1) / threads;

    gamma_kernel_float4<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float4 *>(output.d_data), total_f4);
  }
};

std::unique_ptr<ISPBlock> createGammaCorrectionOptimized() {
  return std::make_unique<GammaCorrectionOptimized>();
}
