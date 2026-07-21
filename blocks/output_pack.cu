#include "blocks.h"
#include "isp_block.h"
#include <cstdio>
#include <memory>
#include <stdexcept>

// ============================================================================
// Output Packing
//
// Converts 3-channel float RGB or YUV [0.0, 1.0] to 3-channel uint8 RGB.
// ============================================================================

__device__ inline void yuv_to_rgb(float y, float u, float v,
                                   uint8_t &r, uint8_t &g, uint8_t &b) {
    u -= 0.5f;
    v -= 0.5f;
    float rf = y + 1.5748f * v;
    float gf = y - 0.1873f * u - 0.4681f * v;
    float bf = y + 1.8556f * u;
    r = static_cast<uint8_t>(fminf(fmaxf(rf, 0.0f), 1.0f) * 255.0f + 0.5f);
    g = static_cast<uint8_t>(fminf(fmaxf(gf, 0.0f), 1.0f) * 255.0f + 0.5f);
    b = static_cast<uint8_t>(fminf(fmaxf(bf, 0.0f), 1.0f) * 255.0f + 0.5f);
}

__global__ void output_pack_yuv_kernel(const float* __restrict__ yuv_float,
                                       uint8_t* __restrict__ rgb_u8,
                                       int num_pixels) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= num_pixels) return;

    int in_idx  = px * 3;
    int out_idx = px * 3;
    yuv_to_rgb(yuv_float[in_idx], yuv_float[in_idx + 1], yuv_float[in_idx + 2],
               rgb_u8[out_idx], rgb_u8[out_idx + 1], rgb_u8[out_idx + 2]);
}

__global__ void output_pack_rgb_kernel(const float* __restrict__ rgb_float,
                                       uint8_t* __restrict__ rgb_u8,
                                       int num_pixels) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= num_pixels) return;

    int idx = px * 3;
    rgb_u8[idx] = static_cast<uint8_t>(fminf(fmaxf(rgb_float[idx], 0.0f), 1.0f) * 255.0f + 0.5f);
    rgb_u8[idx + 1] = static_cast<uint8_t>(fminf(fmaxf(rgb_float[idx + 1], 0.0f), 1.0f) * 255.0f + 0.5f);
    rgb_u8[idx + 2] = static_cast<uint8_t>(fminf(fmaxf(rgb_float[idx + 2], 0.0f), 1.0f) * 255.0f + 0.5f);
}

// --- ISPBlock Implementation ---
class OutputPack : public ISPBlock {
public:
    OutputPack() = default;

    const char* name() const override { return "OutputPack (float->u8)"; }

    void process(const FrameBuffer& input, FrameBuffer& output,
                 cudaStream_t stream) override {
        if ((input.format != PixelFormat::YUV_FLOAT &&
             input.format != PixelFormat::RGB_FLOAT) || input.channels != 3) {
            throw std::invalid_argument(
                "OutputPack requires 3-channel RGB_FLOAT or YUV_FLOAT input");
        }

        // Allocate output: same dimensions, uint8 RGB
        output.width    = input.width;
        output.height   = input.height;
        output.channels = 3;
        output.format   = PixelFormat::RGB_U8;
        output.allocate();

        int num_pixels = input.width * input.height;
        int threads = 256;
        int blocks = (num_pixels + threads - 1) / threads;

        const auto *d_in = static_cast<const float *>(input.d_data);
        auto *d_out = static_cast<uint8_t *>(output.d_data);
        if (input.format == PixelFormat::YUV_FLOAT) {
            output_pack_yuv_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out,
                                                                    num_pixels);
        } else {
            output_pack_rgb_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out,
                                                                    num_pixels);
        }
    }
};

// Factory function (called from main.cpp)
std::unique_ptr<ISPBlock> createOutputPack() {
    return std::make_unique<OutputPack>();
}
