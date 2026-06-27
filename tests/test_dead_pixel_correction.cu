#include <cstdint>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>

#include "blocks.h"
#include "frame_buffer.h"
#include "isp_block.h"

class DeadPixelCorrectionTest : public ::testing::Test {
  protected:
	void SetUp() override { CUDA_CHECK((cudaStreamCreate(&stream_))); }

	void TearDown() override { CUDA_CHECK(cudaStreamDestroy(stream_)); }

	cudaStream_t stream_;
};

// ============================================================================
// Deterministic Correctness Test
// ============================================================================
TEST_F(DeadPixelCorrectionTest, CorrectnessDeterministic) {
	const int width = 128;
	const int height = 128;
	const int bit_depth = 12;

	auto deadPixelCorrection = createDeadPixelCorrection(64, 64);

	// Baseline value is 1000.
	std::vector<uint16_t> h_input(width * height, 1000);

	const int hot_idx = 64 * width + 64;
	const int dead_idx = 64 * width + 70;

	// Set a hot pixel (4000 is > max_val (1000) + th_hot (64))
	h_input[hot_idx] = 4000;
	// Set a dead pixel (0 is < min_val (1000) - th_dead (64))
	h_input[dead_idx] = 0;

	FrameBuffer fb_in;
	fb_in.width = width;
	fb_in.height = height;
	fb_in.channels = 1;
	fb_in.format = PixelFormat::BAYER_RGGB;
	fb_in.bit_depth = bit_depth;
	fb_in.allocate();
	CUDA_CHECK(cudaMemcpy(fb_in.d_data, h_input.data(), fb_in.sizeBytes(),
						  cudaMemcpyHostToDevice));

	FrameBuffer fb_out;
	deadPixelCorrection->process(fb_in, fb_out, stream_);
	CUDA_CHECK(cudaStreamSynchronize(stream_));

	std::vector<uint16_t> h_out(width * height);
	CUDA_CHECK(cudaMemcpy(h_out.data(), fb_out.d_data, fb_out.sizeBytes(),
						  cudaMemcpyDeviceToHost));

	// The hot pixel and dead pixel should be corrected to the average of their neighbors (1000)
	EXPECT_EQ(h_out[hot_idx], 1000);
	EXPECT_EQ(h_out[dead_idx], 1000);

	// All other pixels should remain 1000
	for (int i = 0; i < width * height; ++i) {
		EXPECT_EQ(h_out[i], 1000) << "Mismatch at index " << i;
	}

	fb_in.free();
	fb_out.free();
}

// ============================================================================
// Randomized Stability & Bounds Test
// ============================================================================
TEST_F(DeadPixelCorrectionTest, CorrectnessRandom) {
	const int width = 128;
	const int height = 128;
	const int bit_depth = 12;
	const uint16_t max_val = (1U << bit_depth) - 1;

	auto deadPixelCorrection = createDeadPixelCorrection(64, 64);

	std::vector<uint16_t> h_input(width * height);
	for (int i = 0; i < width * height; ++i) {
		// Generate random values within [1, max_val] to avoid natural 0s
		h_input[i] = 1 + (rand() % max_val);
	}

	const int hot_idx = 64 * width + 64;
	const int dead_idx = 64 * width + 70;

	h_input[hot_idx] = max_val;
	h_input[dead_idx] = 0;

	FrameBuffer fb_in;
	fb_in.width = width;
	fb_in.height = height;
	fb_in.channels = 1;
	fb_in.format = PixelFormat::BAYER_RGGB;
	fb_in.bit_depth = bit_depth;
	fb_in.allocate();
	CUDA_CHECK(cudaMemcpy(fb_in.d_data, h_input.data(), fb_in.sizeBytes(),
						  cudaMemcpyHostToDevice));

	FrameBuffer fb_out;
	deadPixelCorrection->process(fb_in, fb_out, stream_);
	CUDA_CHECK(cudaStreamSynchronize(stream_));

	std::vector<uint16_t> h_out(width * height);
	CUDA_CHECK(cudaMemcpy(h_out.data(), fb_out.d_data, fb_out.sizeBytes(),
						  cudaMemcpyDeviceToHost));

	// Verify corrected values are within valid 12-bit range
	for (int i = 0; i < width * height; ++i) {
		EXPECT_LE(h_out[i], max_val);
	}

	// Verify that the explicitly set dead/hot pixels are corrected and no longer 0 or max_val
	// (Since neighbor values are random in [1, max_val], their average is highly unlikely to be 0 or max_val)
	EXPECT_GT(h_out[hot_idx], 0);
	EXPECT_LT(h_out[hot_idx], max_val);
	EXPECT_GT(h_out[dead_idx], 0);
	EXPECT_LT(h_out[dead_idx], max_val);

	fb_in.free();
	fb_out.free();
}

// ============================================================================
// Performance Test (Benchmark)
// ============================================================================
TEST_F(DeadPixelCorrectionTest, Performance_4K) {
	const int width = 3840;
	const int height = 2160;
	const int bit_depth = 12;
	const int num_iterations = 100;

	auto deadPixelCorrection = createDeadPixelCorrection(64, 64);

	FrameBuffer fb_in;
	fb_in.width = width;
	fb_in.height = height;
	fb_in.channels = 1;
	fb_in.format = PixelFormat::BAYER_RGGB;
	fb_in.bit_depth = bit_depth;
	fb_in.allocate();
	CUDA_CHECK(cudaMemset(fb_in.d_data, 0, fb_in.sizeBytes()));

	FrameBuffer fb_out;
	// Warm up
	deadPixelCorrection->process(fb_in, fb_out, stream_);
	CUDA_CHECK(cudaStreamSynchronize(stream_));

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start, stream_));
	for (int i = 0; i < num_iterations; ++i) {
		deadPixelCorrection->process(fb_in, fb_out, stream_);
	}
	CUDA_CHECK(cudaEventRecord(stop, stream_));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float ms = 0;
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
	ms /= num_iterations;

	// DPC reads 1 uint16_t and writes 1 uint16_t per pixel
	size_t bytes_per_frame = static_cast<size_t>(width) * height * (sizeof(uint16_t) + sizeof(uint16_t));
	float bw = (bytes_per_frame / (ms * 1e-3f)) / (1024 * 1024 * 1024);

	std::cout << "[ PERF     ] Resolution: " << width << "x" << height << "\n";
	std::cout << "[ PERF     ] Avg Time  : " << ms << " ms (" << bw << " GB/s)\n";

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	fb_in.free();
	fb_out.free();
}