#include "blocks.h"
#include "isp_block.h"

#include <memory>
#include <stdexcept>

__global__ void colorCorrectionMatrixKernel(float* rgb, int pixel_count,
                                            ColorCorrectionMatrix matrix) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;

    float* p = rgb + pixel * 3;
    const float r = p[0];
    const float g = p[1];
    const float b = p[2];
    const auto& m = matrix.values;

    p[0] = m[0] * r + m[1] * g + m[2] * b;
    p[1] = m[3] * r + m[4] * g + m[5] * b;
    p[2] = m[6] * r + m[7] * g + m[8] * b;
}

class ColorCorrectionMatrixBlock : public ISPBlock {
public:
    explicit ColorCorrectionMatrixBlock(ColorCorrectionMatrix matrix)
        : matrix_(matrix) {}

    const char* name() const override { return "ColorCorrectionMatrix"; }

    void process(const FrameBuffer& input, FrameBuffer& output,
                 cudaStream_t stream) override {
        if (input.format != PixelFormat::RGB_FLOAT || input.channels != 3) {
            throw std::invalid_argument(
                "ColorCorrectionMatrix requires 3-channel RGB_FLOAT input");
        }

        output = input;
        const int pixel_count = input.width * input.height;
        constexpr int threads = 256;
        const int blocks = (pixel_count + threads - 1) / threads;
        colorCorrectionMatrixKernel<<<blocks, threads, 0, stream>>>(
            static_cast<float*>(output.d_data), pixel_count, matrix_);
    }

private:
    ColorCorrectionMatrix matrix_;
};

std::unique_ptr<ISPBlock> createColorCorrectionMatrix(ColorCorrectionMatrix matrix) {
    return std::make_unique<ColorCorrectionMatrixBlock>(matrix);
}
