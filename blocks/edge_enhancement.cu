#include "blocks.h"
#include "frame_buffer.h"
#include "isp_block.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector_functions.h>

#define PADDING_RADIUS 2
#define TILE_WIDTH 16
#define NEIGHBOR_WINDOW_SIZE (TILE_WIDTH + 2 * PADDING_RADIUS) // 20

__device__ inline float3 rgbToYuv(float r, float g, float b) {
	float3 yuv;
	yuv.x = 0.2126f * r + 0.7152f * g + 0.0722f * b;		 // Y
	yuv.y = -0.1146f * r - 0.3854f * g + 0.5000f * b + 0.5f; // U
	yuv.z = 0.5000f * r - 0.4542f * g - 0.0458f * b + 0.5f;	 // V
	return yuv;
}

__device__ inline float3 yuvToRgb(float3 yuv) {
	float u = yuv.y - 0.5f;
	float v = yuv.z - 0.5f;

	float3 rgb;
	rgb.x = yuv.x + 1.5748f * v;			   // R
	rgb.y = yuv.x - 0.1873f * u - 0.4681f * v; // G
	rgb.z = yuv.x + 1.8556f * u;			   // B

	// Clamp to [0.0f, 1.0f] range
	rgb.x = fmaxf(0.0f, fminf(rgb.x, 1.0f));
	rgb.y = fmaxf(0.0f, fminf(rgb.y, 1.0f));
	rgb.z = fmaxf(0.0f, fminf(rgb.z, 1.0f));
	return rgb;
}

__global__ void edgeEnhancementKernel(const float *input, float *output,
									  int width, int height,
									  EdgeEnhancementConfig config) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	__shared__ float3 sh_yuv[NEIGHBOR_WINDOW_SIZE][NEIGHBOR_WINDOW_SIZE];

	int clamp_x = max(0, min(x, width - 1));
	int clamp_y = max(0, min(y, height - 1));

	int idx = (clamp_x + clamp_y * width) * 3;
	sh_yuv[ty + PADDING_RADIUS][tx + PADDING_RADIUS] =
		make_float3(input[idx], input[idx + 1], input[idx + 2]);

	if (tx < PADDING_RADIUS) {
		int halo_x = max(0, min(x - PADDING_RADIUS, width - 1));
		int halo_idx = (halo_x + clamp_y * width) * 3;
		sh_yuv[ty + PADDING_RADIUS][tx] = make_float3(
			input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}

	if (tx >= blockDim.x - PADDING_RADIUS) {
		int halo_x = max(0, min(x + PADDING_RADIUS, width - 1));
		int halo_idx = (halo_x + clamp_y * width) * 3;
		sh_yuv[ty + PADDING_RADIUS][tx + 2 * PADDING_RADIUS] = make_float3(
			input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}

	if (ty < PADDING_RADIUS) {
		int halo_y = max(0, min(y - PADDING_RADIUS, height - 1));
		int halo_idx = (clamp_x + halo_y * width) * 3;
		sh_yuv[ty][tx + PADDING_RADIUS] = make_float3(
			input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}

	if (ty >= blockDim.y - PADDING_RADIUS) {
		int halo_y = max(0, min(y + PADDING_RADIUS, height - 1));
		int halo_idx = (clamp_x + halo_y * width) * 3;
		sh_yuv[ty + 2 * PADDING_RADIUS][tx + PADDING_RADIUS] = make_float3(
			input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}

	// left-top corner
	if (tx < PADDING_RADIUS && ty < PADDING_RADIUS) {
		int halo_x = max(0, min(x - PADDING_RADIUS, width - 1));
		int halo_y = max(0, min(y - PADDING_RADIUS, height - 1));
		int halo_idx = (halo_x + halo_y * width) * 3;
		sh_yuv[ty][tx] = make_float3(input[halo_idx], input[halo_idx + 1],
									 input[halo_idx + 2]);
	}
	// Top-right corner.
	if (tx >= blockDim.x - PADDING_RADIUS && ty < PADDING_RADIUS) {
		int halo_x = max(0, min(x + PADDING_RADIUS, width - 1));
		int halo_y = max(0, min(y - PADDING_RADIUS, height - 1));
		int halo_idx = (halo_x + halo_y * width) * 3;
		sh_yuv[ty][tx + 2 * PADDING_RADIUS] = make_float3(
			input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}
	// Bottom-left corner.
	if (tx < PADDING_RADIUS && ty >= blockDim.y - PADDING_RADIUS) {
		int halo_x = max(0, min(x - PADDING_RADIUS, width - 1));
		int halo_y = max(0, min(y + PADDING_RADIUS, height - 1));
		int halo_idx = (halo_x + halo_y * width) * 3;
		sh_yuv[ty + 2 * PADDING_RADIUS][tx] = make_float3(
			input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}
	// Bottom-right corner.
	if (tx >= blockDim.x - PADDING_RADIUS &&
		ty >= blockDim.y - PADDING_RADIUS) {
		int halo_x = max(0, min(x + PADDING_RADIUS, width - 1));
		int halo_y = max(0, min(y + PADDING_RADIUS, height - 1));
		int halo_idx = (halo_x + halo_y * width) * 3;
		sh_yuv[ty + 2 * PADDING_RADIUS][tx + 2 * PADDING_RADIUS] = make_float3(
			input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}

	__syncthreads();

	if (x >= width || y >= height) return;

	float sh_yuv_blurred = 0.0f;

	float weighted_sum = 0.0f;
	float blurred_val = 0.0f;
	int center_x = tx + PADDING_RADIUS;
	int center_y = ty + PADDING_RADIUS;
	for (int dy = -PADDING_RADIUS; dy <= PADDING_RADIUS; ++dy) {
		for (int dx = -PADDING_RADIUS; dx <= PADDING_RADIUS; ++dx) {
			blurred_val += sh_yuv[center_y + dy][center_x + dx].x;
			weighted_sum += 1.0f;
		}
	}
	sh_yuv_blurred = blurred_val / weighted_sum;

	float edge = sh_yuv[center_y][center_x].x - sh_yuv_blurred;

	if (fabs(edge) < config.threshold) {
		edge = 0.0f;
	} else {
		float sign = (edge > 0.0f) ? 1.0f : -1.0f;
		edge = sign * (fabs(edge) - config.threshold);
	}

	float enhancement = edge * config.strength;

	enhancement =
		max(-config.clamp_limit, min(enhancement, config.clamp_limit));

	// Enhance luma and keep chroma unchanged. Color conversion is a separate
	// pipeline concern, so bypassing this block never changes the format.
	float out_y = fmaxf(0.0f, fminf(sh_yuv[center_y][center_x].x + enhancement, 1.0f));
	int out_idx = (x + y * width) * 3;
	output[out_idx] = out_y;
	output[out_idx + 1] = sh_yuv[center_y][center_x].y;
	output[out_idx + 2] = sh_yuv[center_y][center_x].z;
}

class EdgeEnhancement : public ISPBlock {
  public:
	EdgeEnhancement(const EdgeEnhancementConfig &config) : config_(config) {
		if (!std::isfinite(config.strength) || config.strength < 0.0f ||
			!std::isfinite(config.threshold) || config.threshold < 0.0f ||
			!std::isfinite(config.clamp_limit) || config.clamp_limit < 0.0f) {
			throw std::invalid_argument("Invalid EdgeEnhancement configuration");
		}
	}

	~EdgeEnhancement() = default;

	const char *name() const override { return "Edge Enhancement"; }

	void process(const FrameBuffer &input, FrameBuffer &output,
				 cudaStream_t stream) override {
		if (input.format != PixelFormat::YUV_FLOAT || input.channels != 3) {
			throw std::invalid_argument(
				"EdgeEnhancement requires 3-channel YUV_FLOAT input");
		}
		if (input.width <= 0 || input.height <= 0 || input.d_data == nullptr) {
			throw std::invalid_argument(
				"EdgeEnhancement requires a non-empty input");
		}

		output.width = input.width;
		output.height = input.height;
		output.channels = 3;
		output.format = PixelFormat::YUV_FLOAT;
		output.allocate();
		int w = input.width;
		int h = input.height;

		dim3 block(TILE_WIDTH, TILE_WIDTH);
		dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

		const auto *d_in = static_cast<const float *>(input.d_data);
		auto *d_out = static_cast<float *>(output.d_data);

		edgeEnhancementKernel<<<grid, block, 0, stream>>>(d_in, d_out, w, h,
														  config_);
	}

  private:
	EdgeEnhancementConfig config_;
};

// ============================================================================
// Factory functions
// ============================================================================
std::unique_ptr<ISPBlock>
createEdgeEnhancement(const EdgeEnhancementConfig &config) {
	return std::make_unique<EdgeEnhancement>(config);
}
