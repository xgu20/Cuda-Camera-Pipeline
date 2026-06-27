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
// Texture-Based Hardware Bilinear Interpolation (Optimized LSC)
// ============================================================================

template <int RED_BX, int RED_BY>
__global__ void applyLensShadingCorrectionKernel(
	uint16_t *d_data, int width, int height, cudaTextureObject_t texObj,
	int grid_width, int grid_height, uint16_t cut_off) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	// Map coordinates
	float u_grid = (width > 1) ? static_cast<float>(x) * (grid_width - 1) / (width - 1) : 0.0f;
	float v_grid = (height > 1) ? static_cast<float>(y) * (grid_height - 1) / (height - 1) : 0.0f;

	// Align to texel center (0.5 offset)
	float u_tex = u_grid + 0.5f;
	float v_tex = v_grid + 0.5f;

	float4 gain4 = tex2D<float4>(texObj, u_tex, v_tex);

	bool is_red = (x & 1) == RED_BX && (y & 1) == RED_BY;
	bool is_blue = (x & 1) == (1 - RED_BX) && (y & 1) == (1 - RED_BY);
	bool is_gr = (x & 1) == (1 - RED_BX) && (y & 1) == RED_BY;

	float gain = is_red ? gain4.x : (is_blue ? gain4.w : (is_gr ? gain4.y : gain4.z));

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

		// Pack R, Gr, Gb, B LUT into float4 array
		const size_t expected_size =
			static_cast<size_t>(grid_width) * grid_height;
		std::vector<float4> h_grid(expected_size);
		for (size_t i = 0; i < expected_size; ++i) {
			h_grid[i] = make_float4(lut[0][i], lut[1][i], lut[2][i], lut[3][i]);
		}

		// Allocate 2D CUDA Array
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		CUDA_CHECK(cudaMallocArray(&d_array_, &channelDesc, grid_width, grid_height));

		// Copy data to CUDA Array
		CUDA_CHECK(cudaMemcpy2DToArray(d_array_, 0, 0, h_grid.data(), grid_width * sizeof(float4),
									   grid_width * sizeof(float4), grid_height,
									   cudaMemcpyHostToDevice));

		// Resource description
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = d_array_;

		// Texture description
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		// Create Texture Object
		CUDA_CHECK(cudaCreateTextureObject(&texObj_, &resDesc, &texDesc, nullptr));
	}

	~LensShadingCorrection() override {
		if (texObj_) {
			cudaDestroyTextureObject(texObj_);
		}
		if (d_array_) {
			cudaFreeArray(d_array_);
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
		d_out, w, h, texObj_, grid_width_, grid_height_, cut_off_)

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

	cudaArray_t d_array_ = nullptr;
	cudaTextureObject_t texObj_ = 0;
};

std::unique_ptr<ISPBlock>
createLensShadingCorrection(const std::vector<std::vector<float>> &lut,
							int grid_width, int grid_height, int bit_depth) {
	return std::make_unique<LensShadingCorrection>(lut, grid_width, grid_height,
												   bit_depth);
}