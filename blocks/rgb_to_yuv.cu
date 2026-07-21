#include "blocks.h"
#include "frame_buffer.h"
#include "isp_block.h"

#include <memory>
#include <stdexcept>

__global__ void rgbToYuvKernel(const float *rgb, float *yuv, int pixels) {
	const int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p >= pixels) return;
	const int i = p * 3;
	const float r = rgb[i];
	const float g = rgb[i + 1];
	const float b = rgb[i + 2];
	yuv[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
	yuv[i + 1] = -0.1146f * r - 0.3854f * g + 0.5000f * b + 0.5f;
	yuv[i + 2] = 0.5000f * r - 0.4542f * g - 0.0458f * b + 0.5f;
}

class RgbToYuv final : public ISPBlock {
  public:
	const char *name() const override { return "RGB to YUV"; }

	void process(const FrameBuffer &input, FrameBuffer &output,
				 cudaStream_t stream) override {
		if (input.format != PixelFormat::RGB_FLOAT || input.channels != 3) {
			throw std::invalid_argument("RgbToYuv requires 3-channel RGB_FLOAT input");
		}
		if (input.width <= 0 || input.height <= 0 || input.d_data == nullptr) {
			throw std::invalid_argument("RgbToYuv requires a non-empty input");
		}
		output.width = input.width;
		output.height = input.height;
		output.channels = 3;
		output.format = PixelFormat::YUV_FLOAT;
		output.allocate();
		const int pixels = input.width * input.height;
		constexpr int threads = 256;
		rgbToYuvKernel<<<(pixels + threads - 1) / threads, threads, 0, stream>>>(
			static_cast<const float *>(input.d_data),
			static_cast<float *>(output.d_data), pixels);
	}
};

std::unique_ptr<ISPBlock> createRgbToYuv() {
	return std::make_unique<RgbToYuv>();
}
