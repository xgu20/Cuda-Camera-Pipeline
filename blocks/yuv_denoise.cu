#include "blocks.h"
#include "frame_buffer.h"
#include "isp_block.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector_functions.h>
#include <vector_types.h>

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

__global__ void yuvDenoiseKernel(const float *input, float *output, int width,
								 int height, const YuvDenoiseConfig config) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = blockIdx.x * blockDim.x + tx;
	int y = blockIdx.y * blockDim.y + ty;

	// Shared-memory tile.
	__shared__ float3 sh_yuv[NEIGHBOR_WINDOW_SIZE][NEIGHBOR_WINDOW_SIZE];

	// Clamp coordinates so every cooperative load remains in bounds.
	int clamp_x = max(0, min(x, width - 1));
	int clamp_y = max(0, min(y, height - 1));

	// 1. Load the center tile. Out-of-range threads still load clamped pixels.
	int idx = (clamp_x + clamp_y * width) * 3;
	sh_yuv[ty + PADDING_RADIUS][tx + PADDING_RADIUS] =
		make_float3(input[idx], input[idx + 1], input[idx + 2]);

	// 2. Load the four edge halos.
	// Left halo.
	if (tx < PADDING_RADIUS) {
		int halo_x = max(0, x - PADDING_RADIUS);
		int halo_idx = (halo_x + clamp_y * width) * 3;
		sh_yuv[ty + PADDING_RADIUS][tx] =
			make_float3(input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}
	// Right halo.
	if (tx >= blockDim.x - PADDING_RADIUS) {
		int halo_x = min(width - 1, x + PADDING_RADIUS);
		int halo_idx = (halo_x + clamp_y * width) * 3;
		sh_yuv[ty + PADDING_RADIUS][tx + 2 * PADDING_RADIUS] =
			make_float3(input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}
	// Top halo.
	if (ty < PADDING_RADIUS) {
		int halo_y = max(0, y - PADDING_RADIUS);
		int halo_idx = (clamp_x + halo_y * width) * 3;
		sh_yuv[ty][tx + PADDING_RADIUS] =
			make_float3(input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}
	// Bottom halo.
	if (ty >= blockDim.y - PADDING_RADIUS) {
		int halo_y = min(height - 1, y + PADDING_RADIUS);
		int halo_idx = (clamp_x + halo_y * width) * 3;
		sh_yuv[ty + 2 * PADDING_RADIUS][tx + PADDING_RADIUS] =
			make_float3(input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}

	// 3. Load the four corners.
	// Top-left corner.
	if (tx < PADDING_RADIUS && ty < PADDING_RADIUS) {
		int halo_x = max(0, x - PADDING_RADIUS);
		int halo_y = max(0, y - PADDING_RADIUS);
		int halo_idx = (halo_x + halo_y * width) * 3;
		sh_yuv[ty][tx] =
			make_float3(input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}
	// Top-right corner.
	if (tx >= blockDim.x - PADDING_RADIUS && ty < PADDING_RADIUS) {
		int halo_x = min(width - 1, x + PADDING_RADIUS);
		int halo_y = max(0, y - PADDING_RADIUS);
		int halo_idx = (halo_x + halo_y * width) * 3;
		sh_yuv[ty][tx + 2 * PADDING_RADIUS] =
			make_float3(input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}
	// Bottom-left corner.
	if (tx < PADDING_RADIUS && ty >= blockDim.y - PADDING_RADIUS) {
		int halo_x = max(0, x - PADDING_RADIUS);
		int halo_y = min(height - 1, y + PADDING_RADIUS);
		int halo_idx = (halo_x + halo_y * width) * 3;
		sh_yuv[ty + 2 * PADDING_RADIUS][tx] =
			make_float3(input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}
	// Bottom-right corner.
	if (tx >= blockDim.x - PADDING_RADIUS &&
		ty >= blockDim.y - PADDING_RADIUS) {
		int halo_x = min(width - 1, x + PADDING_RADIUS);
		int halo_y = min(height - 1, y + PADDING_RADIUS);
		int halo_idx = (halo_x + halo_y * width) * 3;
		sh_yuv[ty + 2 * PADDING_RADIUS][tx + 2 * PADDING_RADIUS] =
			make_float3(input[halo_idx], input[halo_idx + 1], input[halo_idx + 2]);
	}

	__shared__ float sh_spatial[2 * PADDING_RADIUS + 1][2 * PADDING_RADIUS + 1];
	if (ty < 2 * PADDING_RADIUS + 1 && tx < 2 * PADDING_RADIUS + 1) {
		float spatial_var2 =
			2.0f * config.spatial_sigma * config.spatial_sigma + 1e-6f;
		int dy = ty - PADDING_RADIUS;
		int dx = tx - PADDING_RADIUS;
		sh_spatial[ty][tx] = __expf(-(dy * dy + dx * dx) / spatial_var2);
	}

	// Ensure all cooperative loads are complete.
	__syncthreads();

	// Out-of-range threads can return after the barrier.
	if (x >= width || y >= height)
		return;

	// Read the shared tile and evaluate the bilateral filter.
	float3 center_yuv = sh_yuv[ty + PADDING_RADIUS][tx + PADDING_RADIUS];

	float sum_y = 0.0f;
	float weight_sum_y = 0.0f;

	float sum_u = 0.0f;
	float sum_v = 0.0f;
	float weight_sum_chroma = 0.0f;

	float luma_var2 =
		2.0f * config.luma_range_sigma * config.luma_range_sigma + 1e-6f;
	float chroma_var2 =
		2.0f * config.chroma_range_sigma * config.chroma_range_sigma + 1e-6f;

	// Pre-calculate merged scale for __exp2f (log2(e) = 1.44269504f)
	float luma_exp_scale = -1.44269504f / luma_var2;
	float chroma_exp_scale = -1.44269504f / chroma_var2;

#pragma unroll
	for (int dy = -PADDING_RADIUS; dy <= PADDING_RADIUS; ++dy) {
#pragma unroll
		for (int dx = -PADDING_RADIUS; dx <= PADDING_RADIUS; ++dx) {
			float3 neighbor_yuv =
				sh_yuv[ty + PADDING_RADIUS + dy][tx + PADDING_RADIUS + dx];

			// 1. Spatial distance and weight
			float spacial_w = sh_spatial[dy + PADDING_RADIUS][dx + PADDING_RADIUS];

			// 2. Luma Difference
			float diff_y2 = (neighbor_yuv.x - center_yuv.x) *
							(neighbor_yuv.x - center_yuv.x);

			// 3. Luma Range Weight (using fast base-2 exp)
			float luma_range_w = exp2f(diff_y2 * luma_exp_scale);
			float weight_y = spacial_w * luma_range_w;

			// 4. Chroma Range Weight - Joint Bilateral
			float chroma_range_w = exp2f(diff_y2 * chroma_exp_scale);
			float weight_chroma = spacial_w * chroma_range_w;

			// 5. Sum and weight_sum
			sum_y += weight_y * neighbor_yuv.x;
			weight_sum_y += weight_y;

			sum_u += neighbor_yuv.y * weight_chroma;
			sum_v += neighbor_yuv.z * weight_chroma;
			weight_sum_chroma += weight_chroma;
		}
	}

	// 6. Normalization
	// The center sample always has a positive weight, so these denominators
	// cannot be zero for a valid configuration.  Adding epsilon here biased a
	// perfectly constant image slightly darker on every invocation.
	float filtered_y = sum_y / weight_sum_y;
	float filtered_u = sum_u / weight_sum_chroma;
	float filtered_v = sum_v / weight_sum_chroma;

	// 7. alpha blending
	float final_y = (1.0f - config.luma_strength) * center_yuv.x +
					config.luma_strength * filtered_y;
	float final_u = (1.0f - config.chroma_strength) * center_yuv.y +
					config.chroma_strength * filtered_u;
	float final_v = (1.0f - config.chroma_strength) * center_yuv.z +
					config.chroma_strength * filtered_v;

	// 8. write YUV directly — no conversion back to RGB
	int out_idx = (y * width + x) * 3;
	output[out_idx]     = final_y;
	output[out_idx + 1] = final_u;
	output[out_idx + 2] = final_v;
}

class YuvDenoise : public ISPBlock {
  public:
	YuvDenoise(const YuvDenoiseConfig &config) : config_(config) {
		if (!std::isfinite(config.spatial_sigma) || config.spatial_sigma <= 0.0f ||
			!std::isfinite(config.luma_range_sigma) ||
			config.luma_range_sigma <= 0.0f ||
			!std::isfinite(config.chroma_range_sigma) ||
			config.chroma_range_sigma <= 0.0f ||
			!std::isfinite(config.luma_strength) || config.luma_strength < 0.0f ||
			config.luma_strength > 1.0f ||
			!std::isfinite(config.chroma_strength) ||
			config.chroma_strength < 0.0f || config.chroma_strength > 1.0f) {
			throw std::invalid_argument("Invalid YuvDenoise configuration");
		}
	}

	~YuvDenoise() {}

	const char *name() const override { return "YuvDenoise"; }

	void process(const FrameBuffer &input, FrameBuffer &output,
				 cudaStream_t stream) override {
		if (input.format != PixelFormat::YUV_FLOAT || input.channels != 3) {
			throw std::invalid_argument(
				"YuvDenoise requires 3-channel YUV_FLOAT input");
		}
		if (input.width <= 0 || input.height <= 0 || input.d_data == nullptr) {
			throw std::invalid_argument("YuvDenoise requires a non-empty input");
		}

		output.width = input.width;
		output.height = input.height;
		output.channels = input.channels;
		output.format = PixelFormat::YUV_FLOAT;
		output.allocate();

		int w = input.width;
		int h = input.height;

		dim3 block(TILE_WIDTH, TILE_WIDTH);
		dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

		const auto *d_in = static_cast<const float *>(input.d_data);
		auto *d_out = static_cast<float *>(output.d_data);

		yuvDenoiseKernel<<<grid, block, 0, stream>>>(d_in, d_out, w, h, config_);
	}

  private:
	YuvDenoiseConfig config_;
};

// ============================================================================
// Factory functions
// ============================================================================
std::unique_ptr<ISPBlock> createYuvDenoise(const YuvDenoiseConfig &config) {
	return std::make_unique<YuvDenoise>(config);
}
