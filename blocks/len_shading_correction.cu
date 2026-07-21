#include "blocks.h"
#include "frame_buffer.h"
#include "isp_block.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// ============================================================================
// Manual Bilinear Interpolation (High-Precision FP32 LSC, Fused Channel Reads)
// ============================================================================

template <int RED_BX, int RED_BY>
__global__ void applyLensShadingCorrectionKernel(
	uint16_t *d_data, int width, int height, const float *d_lut,
	int grid_width, int grid_height, uint16_t cut_off) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	// Map coordinates to grid floating point indices
	float u_grid = (width > 1) ? static_cast<float>(x) * (grid_width - 1) / (width - 1) : 0.0f;
	float v_grid = (height > 1) ? static_cast<float>(y) * (grid_height - 1) / (height - 1) : 0.0f;

	// Bounding cell indices
	int x0 = __float2int_rd(u_grid);
	int y0 = __float2int_rd(v_grid);

	// Clamp indices to ensure we stay within bounds [0, grid_width - 2] & [0, grid_height - 2]
	x0 = max(0, min(x0, grid_width - 2));
	y0 = max(0, min(y0, grid_height - 2));

	int x1 = x0 + 1;
	int y1 = y0 + 1;

	// Interpolation weights in [0.0f, 1.0f]
	float tx = u_grid - x0;
	float ty = v_grid - y0;
	tx = max(0.0f, min(tx, 1.0f));
	ty = max(0.0f, min(ty, 1.0f));

	// Determine the Bayer channel index for the current pixel
	// 0 = R, 1 = Gr, 2 = Gb, 3 = B
	int c = 0;
	bool is_red = (x & 1) == RED_BX && (y & 1) == RED_BY;
	bool is_gr = (x & 1) == (1 - RED_BX) && (y & 1) == RED_BY;
	bool is_gb = (x & 1) == RED_BX && (y & 1) == (1 - RED_BY);

	if (is_red) {
		c = 0;
	} else if (is_gr) {
		c = 1;
	} else if (is_gb) {
		c = 2;
	} else {
		c = 3;
	}

	// Read only the 4 grid corner points for this specific channel
	const float* chan_lut = d_lut + c * (grid_width * grid_height);
	float g00 = chan_lut[x0 + y0 * grid_width];
	float g10 = chan_lut[x1 + y0 * grid_width];
	float g01 = chan_lut[x0 + y1 * grid_width];
	float g11 = chan_lut[x1 + y1 * grid_width];

	// Bilinear interpolation math
	float w00 = (1.0f - tx) * (1.0f - ty);
	float w10 = tx * (1.0f - ty);
	float w01 = (1.0f - tx) * ty;
	float w11 = tx * ty;

	float gain = w00 * g00 + w10 * g10 + w01 * g01 + w11 * g11;

	float val = d_data[x + y * width] * gain;

	d_data[x + y * width] = val > cut_off ? cut_off : (uint16_t)val;
}

class LensShadingCorrection : public ISPBlock {
  public:
	LensShadingCorrection(const std::vector<std::vector<float>> &lut,
						  int grid_width, int grid_height, int bit_depth)
		: grid_width_(grid_width), grid_height_(grid_height),
		  bit_depth_(bit_depth) {
		cut_off_ = (1U << bit_depth_) - 1;
		if (lut.size() != 4) {
			throw std::invalid_argument(
				"LSC LUT must contain exactly 4 channels (R, Gr, Gb, B)");
		}
		if (grid_width_ < 2 || grid_height_ < 2) {
			throw std::invalid_argument(
				"grid_width or grid_height must be larger than 2");
		}

		const size_t grid_sz = static_cast<size_t>(grid_width) * grid_height;

		// Allocate Flat Device Memory for 4 channels
		CUDA_CHECK(cudaMalloc(&d_lut_, 4 * grid_sz * sizeof(float)));

		// Copy each channel's data sequentially to device memory
		for (int c = 0; c < 4; ++c) {
			if (lut[c].size() != grid_sz) {
				throw std::invalid_argument("LSC LUT channel size mismatch");
			}
			CUDA_CHECK(cudaMemcpy(d_lut_ + c * grid_sz, lut[c].data(),
								  grid_sz * sizeof(float), cudaMemcpyHostToDevice));
		}
	}

	~LensShadingCorrection() override {
		if (d_lut_) {
			cudaFree(d_lut_);
		}
	}

	const char *name() const override { return "LensShadingCorrection"; }

	void process(const FrameBuffer &input, FrameBuffer &output,
				 cudaStream_t stream) override {
		output = input;

		auto *d_out = static_cast<uint16_t *>(output.d_data);
		int w = input.width;
		int h = input.height;
		dim3 block(16, 16);
		dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
		if (!input.isBayer()) {
			throw std::invalid_argument("Input data must be bayer");
		}
#define LAUNCH_LSC(RX, RY)                                                     \
	applyLensShadingCorrectionKernel<RX, RY><<<grid, block, 0, stream>>>( \
		d_out, w, h, d_lut_, grid_width_, grid_height_, cut_off_)

		switch (input.format) {
		case PixelFormat::BAYER_RGGB:
			LAUNCH_LSC(0, 0);
			break;
		case PixelFormat::BAYER_BGGR:
			LAUNCH_LSC(1, 1);
			break;
		case PixelFormat::BAYER_GRBG:
			LAUNCH_LSC(1, 0);
			break;
		case PixelFormat::BAYER_GBRG:
			LAUNCH_LSC(0, 1);
			break;
		default:
			throw std::runtime_error("Unsupported Bayer pattern format in "
									 "LensShadingCorrection");
		}

#undef LAUNCH_LSC
	}

  private:
	int grid_width_;
	int grid_height_;
	int bit_depth_;
	uint16_t cut_off_;

	float *d_lut_ = nullptr;
};

std::unique_ptr<ISPBlock>
createLensShadingCorrection(const std::vector<std::vector<float>> &lut,
							int grid_width, int grid_height, int bit_depth) {
	return std::make_unique<LensShadingCorrection>(lut, grid_width, grid_height,
												   bit_depth);
}