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
                           uint16_t bl00, uint16_t bl01,
                           uint16_t bl10, uint16_t bl11) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint16_t bl = ((y & 1) == 0) ? (((x & 1) == 0) ? bl00 : bl01)
                                 : (((x & 1) == 0) ? bl10 : bl11);
    int val = static_cast<int>(data[idx]) - static_cast<int>(bl);
    data[idx] = static_cast<uint16_t>(val > 0 ? val : 0);
}

// --- ISPBlock Implementation ---
class BlackLevelCorrection : public ISPBlock {
public:
    explicit BlackLevelCorrection(const std::vector<uint16_t>& black_levels, PixelFormat format)
        : black_levels_(black_levels), format_(format) {
        if (black_levels.size() != 4) {
            throw std::invalid_argument("BlackLevelCorrection requires 4 black levels (R, Gr, Gb, B)");
        }
    }

    const char* name() const override { return "BlackLevelCorrection"; }

    void process(const FrameBuffer& input, FrameBuffer& output,
                 cudaStream_t stream) override {
        if (!input.isBayer() || input.packing != PixelPacking::UNPACKED_U16) {
            throw std::invalid_argument(
                "BlackLevelCorrection requires unpacked uint16 Bayer input");
        }

        // Arrange black levels to match the 2x2 Bayer pattern grid (y, x)
        // input black_levels: {R, Gr, Gb, B}
        uint16_t R = black_levels_[0];
        uint16_t Gr = black_levels_[1];
        uint16_t Gb = black_levels_[2];
        uint16_t B = black_levels_[3];

        uint16_t bl00, bl01, bl10, bl11;
        if (format_ == PixelFormat::BAYER_RGGB) {
            bl00 = R;  bl01 = Gr;
            bl10 = Gb; bl11 = B;
        } else if (format_ == PixelFormat::BAYER_BGGR) {
            bl00 = B;  bl01 = Gb;
            bl10 = Gr; bl11 = R;
        } else if (format_ == PixelFormat::BAYER_GRBG) {
            bl00 = Gr; bl01 = R;
            bl10 = B;  bl11 = Gb;
        } else { // BAYER_GBRG
            bl00 = Gb; bl01 = B;
            bl10 = R;  bl11 = Gr;
        }

        // In-place operation — output aliases input
        output = input;

        dim3 block(16, 16);
        dim3 grid((input.width + block.x - 1) / block.x,
                  (input.height + block.y - 1) / block.y);

        blc_kernel<<<grid, block, 0, stream>>>(
            static_cast<uint16_t*>(output.d_data),
            input.width, input.height, bl00, bl01, bl10, bl11);
    }

private:
    std::vector<uint16_t> black_levels_;
    PixelFormat format_;
};

// Factory function (called from main.cpp)
std::unique_ptr<ISPBlock> createBlackLevelCorrection(const std::vector<uint16_t>& black_levels, PixelFormat format) {
    return std::make_unique<BlackLevelCorrection>(black_levels, format);
}
