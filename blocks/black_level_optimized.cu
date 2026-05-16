#include "blocks.h"
#include "isp_block.h"
#include <cstdio>
#include <memory>
#include <algorithm>

// ============================================================================
// Black Level Correction (Optimized)
//
// Uses vectorized memory access (uint32_t) and a 1D grid-stride loop to
// maximize memory bandwidth utilization.
// ============================================================================

// --- Optimized CUDA Kernel ---
__global__ void blc_kernel_optimized(uint32_t* data, int num_uint32,
                                     uint32_t black_level_x2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < num_uint32; i += stride) {
        // Read 4 bytes (two 16-bit pixels) at once
        uint32_t val2 = data[i];

        // Unpack into two 32-bit integers and subtract
        int p0 = (val2 & 0xFFFF) - (black_level_x2 & 0xFFFF);
        int p1 = (val2 >> 16) - (black_level_x2 >> 16);

        // Clamp to 0
        p0 = p0 > 0 ? p0 : 0;
        p1 = p1 > 0 ? p1 : 0;

        // Repack and write 4 bytes
        data[i] = (p0 & 0xFFFF) | (p1 << 16);
    }
}

// Tail kernel for the last pixel if the total number of pixels is odd
__global__ void blc_kernel_tail(uint16_t* data, int total_pixels,
                                uint16_t black_level) {
    int val = static_cast<int>(data[total_pixels - 1]) - static_cast<int>(black_level);
    data[total_pixels - 1] = static_cast<uint16_t>(val > 0 ? val : 0);
}

// --- ISPBlock Implementation ---
class BlackLevelCorrectionOptimized : public ISPBlock {
public:
    explicit BlackLevelCorrectionOptimized(uint16_t black_level = 64)
        : black_level_(black_level) {}

    const char* name() const override { return "BlackLevelCorrectionOptimized"; }

    void process(const FrameBuffer& input, FrameBuffer& output,
                 cudaStream_t stream) override {
        if (!input.isBayer()) {
            fprintf(stderr, "[BLC Opt] Warning: expected Bayer input\n");
        }

        // In-place operation — output aliases input
        output = input;

        int total_pixels = input.width * input.height;
        int num_uint32 = total_pixels / 2;
        int remainder = total_pixels % 2;

        // Pack the 16-bit black level into both halves of a 32-bit integer
        uint32_t black_level_x2 = (static_cast<uint32_t>(black_level_) << 16) | black_level_;

        // Use a 1D grid for the grid-stride loop.
        // 256 threads per block is standard.
        int blockSize = 256;
        // Launch enough blocks to saturate the GPU, but cap it to avoid excessive blocks.
        // Max grid size limit here isn't strictly necessary for correctness due to grid-stride,
        // but helps with scheduling efficiency.
        int gridSize = std::min((num_uint32 + blockSize - 1) / blockSize, 65535);

        if (num_uint32 > 0) {
            blc_kernel_optimized<<<gridSize, blockSize, 0, stream>>>(
                reinterpret_cast<uint32_t*>(output.d_data),
                num_uint32, black_level_x2);
        }

        // Handle the potential odd pixel at the end
        if (remainder > 0) {
            blc_kernel_tail<<<1, 1, 0, stream>>>(
                static_cast<uint16_t*>(output.d_data),
                total_pixels, black_level_);
        }
    }

private:
    uint16_t black_level_;
};

// Factory function
std::unique_ptr<ISPBlock> createBlackLevelCorrectionOptimized(uint16_t black_level) {
    return std::make_unique<BlackLevelCorrectionOptimized>(black_level);
}
