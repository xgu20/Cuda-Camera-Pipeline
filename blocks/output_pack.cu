#include "blocks.h"
#include "isp_block.h"
#include <cstdio>
#include <memory>
#include <stdexcept>

// ============================================================================
// Output Packing
//
// Converts 3-channel float RGB [0.0, 1.0] to 3-channel uint8 RGB [0, 255].
// This is the final block in the pipeline before saving to disk.
// ============================================================================

__global__ void output_pack_kernel(const float* __restrict__ rgb_float,
                                   uint8_t* __restrict__ rgb_u8,
                                   int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float val = rgb_float[idx];
    // Clamp to [0, 1] and convert to [0, 255]
    val = fminf(fmaxf(val, 0.0f), 1.0f);
    rgb_u8[idx] = static_cast<uint8_t>(val * 255.0f + 0.5f);
}

// --- ISPBlock Implementation ---
class OutputPack : public ISPBlock {
public:
    OutputPack() = default;

    const char* name() const override { return "OutputPack (float->u8)"; }

    void process(const FrameBuffer& input, FrameBuffer& output,
                 cudaStream_t stream) override {
        if (input.format != PixelFormat::RGB_FLOAT) {
            throw std::invalid_argument("OutputPack requires RGB_FLOAT input");
        }

        // Allocate output: same dimensions, uint8
        output.width    = input.width;
        output.height   = input.height;
        output.channels = 3;
        output.format   = PixelFormat::RGB_U8;
        output.allocate();

        int total = input.width * input.height * input.channels;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        output_pack_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const float*>(input.d_data),
            static_cast<uint8_t*>(output.d_data),
            total);
    }
};

// Factory function (called from main.cpp)
std::unique_ptr<ISPBlock> createOutputPack() {
    return std::make_unique<OutputPack>();
}
