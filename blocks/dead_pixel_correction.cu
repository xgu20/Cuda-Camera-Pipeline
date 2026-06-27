#include "blocks.h"
#include "isp_block.h"
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>

#define PADDING_RADIUS 2
#define TILE_WIDTH 16
#define NEIGHBOR_WINDOW_SIZE (TILE_WIDTH + 2 * PADDING_RADIUS) // 20

__global__ void dpcKernel(const uint16_t *in_data, uint16_t *out_data, int width, int height,
						  uint16_t th_hot, uint16_t th_dead) {
	__shared__ uint16_t neighbor[NEIGHBOR_WINDOW_SIZE][NEIGHBOR_WINDOW_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	const int tid = threadIdx.y * blockDim.x + threadIdx.x;

	const int bx_g = blockIdx.x * TILE_WIDTH; 
	const int by_g = blockIdx.y * TILE_WIDTH; 

	// Cooperatively load tiles and halo pixels into shared memory
	for (int i = tid; i < NEIGHBOR_WINDOW_SIZE * NEIGHBOR_WINDOW_SIZE;
		 i += blockDim.x * blockDim.y) {
		const int sy = i / NEIGHBOR_WINDOW_SIZE;
		const int sx = i % NEIGHBOR_WINDOW_SIZE;

		int gx = bx_g - PADDING_RADIUS + sx;
		int gy = by_g - PADDING_RADIUS + sy;

		// Clamp-to-edge boundary padding to avoid polluting min_val/max_val with zeros
		int clamp_x = max(0, min(width - 1, gx));
		int clamp_y = max(0, min(height - 1, gy));
		neighbor[sy][sx] = in_data[clamp_y * width + clamp_x];
	}
	__syncthreads();

	// Active pixel coordinates for the current thread
	const int gx_thread = bx_g + tx;
	const int gy_thread = by_g + ty;

	// Out-of-bounds threads must not write to global memory, but they had to participate in loading shared memory.
	if (gx_thread >= width || gy_thread >= height) {
		return;
	}

	const int sx_thread = tx + PADDING_RADIUS;
	const int sy_thread = ty + PADDING_RADIUS;

	uint16_t p00 = neighbor[sy_thread][sx_thread];

	// Extract the 8 same-color neighbors in the 5x5 window (spaced by 2 pixels)
	uint16_t pt  = neighbor[sy_thread - 2][sx_thread];
	uint16_t pb  = neighbor[sy_thread + 2][sx_thread];
	uint16_t pl  = neighbor[sy_thread][sx_thread - 2];
	uint16_t pr  = neighbor[sy_thread][sx_thread + 2];
	uint16_t ptl = neighbor[sy_thread - 2][sx_thread - 2];
	uint16_t ptr = neighbor[sy_thread - 2][sx_thread + 2]; // Corrected x offset to + 2
	uint16_t pbl = neighbor[sy_thread + 2][sx_thread - 2];
	uint16_t pbr = neighbor[sy_thread + 2][sx_thread + 2];

	uint16_t neighbors[8] = {pt, pb, pl, pr, ptl, ptr, pbl, pbr};
	uint16_t max_val = pt, min_val = pt;
	for (int i = 1; i < 8; ++i) {
		max_val = max_val > neighbors[i] ? max_val : neighbors[i];
		min_val = min_val < neighbors[i] ? min_val : neighbors[i];
	}

	uint16_t corrected_val = p00;

	// Hot pixel and dead pixel detection
	if (p00 > max_val && (p00 - max_val) > th_hot) {
		corrected_val = static_cast<uint16_t>((static_cast<uint32_t>(pt) + pb + pl + pr) / 4);
	} else if (p00 < min_val && (min_val - p00) > th_dead) {
		corrected_val = static_cast<uint16_t>((static_cast<uint32_t>(pt) + pb + pl + pr) / 4);
	}

	// Write back to the output global memory buffer
	out_data[gy_thread * width + gx_thread] = corrected_val;
}

class DeadPixelCorrection : public ISPBlock {
  public:
	DeadPixelCorrection(uint16_t th_hot, uint16_t th_dead)
		: th_hot_(th_hot), th_dead_(th_dead) {}
	~DeadPixelCorrection() override {}

	const char *name() const override { return "DeadPixelCorrection"; }

	void process(const FrameBuffer &input, FrameBuffer &output,
				 cudaStream_t stream) override {
		if (!input.isBayer() || input.packing != PixelPacking::UNPACKED_U16) {
			throw std::invalid_argument(
				"DeadPixelCorrection requires unpacked uint16 Bayer input");
		}

		// Configure output format and dimensions, then allocate (out-of-place execution)
		output.width = input.width;
		output.height = input.height;
		output.channels = input.channels;
		output.format = input.format;
		output.packing = input.packing;
		output.bit_depth = input.bit_depth;
		output.allocate();

		const uint16_t *in_data = reinterpret_cast<const uint16_t *>(input.d_data);
		uint16_t *out_data = reinterpret_cast<uint16_t *>(output.d_data);

		dim3 block(TILE_WIDTH, TILE_WIDTH);
		dim3 grid((input.width + block.x - 1) / block.x,
				  (input.height + block.y - 1) / block.y);

		dpcKernel<<<grid, block, 0, stream>>>(in_data, out_data, input.width, input.height,
											  th_hot_, th_dead_);
	}

  private:
	uint16_t th_hot_;
	uint16_t th_dead_;
};

// ============================================================================
// Factory functions
// ============================================================================
std::unique_ptr<ISPBlock> createDeadPixelCorrection(uint16_t th_hot,
													uint16_t th_dead) {
	return std::make_unique<DeadPixelCorrection>(th_hot, th_dead);
}