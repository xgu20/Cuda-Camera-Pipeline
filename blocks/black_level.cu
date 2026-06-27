#include "blocks.h"
#include "isp_block.h"
#include <cstdio>
#include <memory>
#include <stdexcept>

// ============================================================================
// Black Level Correction (BLC)
//
// Subtracts a fixed black level offset from every pixel in the Bayer image.
// Clamps to zero to avoid underflow. Operates in-place on uint16_t data.
// ============================================================================

// --- CUDA Kernel ---
__global__ void blc_kernel(uint16_t* data, int width, int height,
                           uint16_t black_level) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int val = static_cast<int>(data[idx]) - static_cast<int>(black_level);
    data[idx] = static_cast<uint16_t>(val > 0 ? val : 0);
}

// --- ISPBlock Implementation ---
class BlackLevelCorrection : public ISPBlock {
public:
    explicit BlackLevelCorrection(uint16_t black_level = 64)
        : black_level_(black_level) {}

    const char* name() const override { return "BlackLevelCorrection"; }

    void process(const FrameBuffer& input, FrameBuffer& output,
                 cudaStream_t stream) override {
        if (!input.isBayer() || input.packing != PixelPacking::UNPACKED_U16) {
            throw std::invalid_argument(
                "BlackLevelCorrection requires unpacked uint16 Bayer input");
        }

        // In-place operation — output aliases input
        output = input;

        dim3 block(16, 16);
        dim3 grid((input.width + block.x - 1) / block.x,
                  (input.height + block.y - 1) / block.y);

        blc_kernel<<<grid, block, 0, stream>>>(
            static_cast<uint16_t*>(output.d_data),
            input.width, input.height, black_level_);
    }

private:
    uint16_t black_level_;
};

// Factory function (called from main.cpp)
std::unique_ptr<ISPBlock> createBlackLevelCorrection(uint16_t black_level) {
    return std::make_unique<BlackLevelCorrection>(black_level);
}
