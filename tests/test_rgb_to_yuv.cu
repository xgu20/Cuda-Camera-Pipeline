#include "blocks.h"
#include "frame_buffer.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

TEST(RgbToYuvTest, ConvertsNonNeutralPixelsAndSetsFormat) {
	constexpr int w = 3, h = 1;
	const std::vector<float> input_data = {
		0.8f, 0.3f, 0.1f,
		0.5f, 0.5f, 0.5f,
		0.0f, 0.0f, 0.0f,
	};
	FrameBuffer input;
	input.width = w;
	input.height = h;
	input.channels = 3;
	input.format = PixelFormat::RGB_FLOAT;
	input.allocate();
	CUDA_CHECK(cudaMemcpy(input.d_data, input_data.data(), input.sizeBytes(),
						  cudaMemcpyHostToDevice));

	FrameBuffer output;
	auto block = createRgbToYuv();
	block->process(input, output, nullptr);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	std::vector<float> actual(input_data.size());
	CUDA_CHECK(cudaMemcpy(actual.data(), output.d_data, output.sizeBytes(),
						  cudaMemcpyDeviceToHost));

	EXPECT_EQ(output.format, PixelFormat::YUV_FLOAT);
	EXPECT_NEAR(actual[0], 0.2126f * 0.8f + 0.7152f * 0.3f + 0.0722f * 0.1f,
				2e-6f);
	EXPECT_NEAR(actual[1], -0.1146f * 0.8f - 0.3854f * 0.3f + 0.5f * 0.1f + 0.5f,
				2e-6f);
	EXPECT_NEAR(actual[2], 0.5f * 0.8f - 0.4542f * 0.3f - 0.0458f * 0.1f + 0.5f,
				2e-6f);
	EXPECT_NEAR(actual[3], 0.5f, 2e-6f);
	EXPECT_NEAR(actual[4], 0.5f, 2e-6f);
	EXPECT_NEAR(actual[5], 0.5f, 2e-6f);

	input.free();
	output.free();
}
