#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "blocks.h"
#include "frame_buffer.h"
#include "isp_pipeline.h"

namespace {

std::vector<uint16_t> download(const FrameBuffer& frame) {
    std::vector<uint16_t> host(static_cast<size_t>(frame.width) * frame.height);
    CUDA_CHECK(cudaMemcpy(host.data(), frame.d_data,
                          host.size() * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    return host;
}

}  // namespace

TEST(ISPPipelineTest, DefaultExecuteIsZeroCopyAndConsumesInput) {
    constexpr int width = 4;
    constexpr int height = 4;
    constexpr uint16_t black_level = 64;

    std::vector<uint16_t> original(width * height);
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = static_cast<uint16_t>(100 + i);
    }

    FrameBuffer input;
    input.width = width;
    input.height = height;
    input.channels = 1;
    input.format = PixelFormat::BAYER_RGGB;
    input.packing = PixelPacking::UNPACKED_U16;
    input.bit_depth = 10;
    input.allocate();
    CUDA_CHECK(cudaMemcpy(input.d_data, original.data(), input.sizeBytes(),
                          cudaMemcpyHostToDevice));

    ISPPipeline pipeline;
    pipeline.addBlock(createBlackLevelCorrection(black_level));

    const FrameBuffer result = pipeline.execute(input);
    const auto input_after = download(input);

    EXPECT_EQ(result.d_data, input.d_data);
    for (size_t i = 0; i < input_after.size(); ++i) {
        EXPECT_EQ(input_after[i], original[i] - black_level);
    }

    input.free();
}

TEST(ISPPipelineTest, PreservingExecuteKeepsInputAndIsRepeatable) {
    constexpr int width = 4;
    constexpr int height = 4;
    constexpr uint16_t black_level = 64;

    std::vector<uint16_t> original(width * height);
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = static_cast<uint16_t>(100 + i);
    }

    FrameBuffer input;
    input.width = width;
    input.height = height;
    input.channels = 1;
    input.format = PixelFormat::BAYER_RGGB;
    input.packing = PixelPacking::UNPACKED_U16;
    input.bit_depth = 10;
    input.allocate();
    CUDA_CHECK(cudaMemcpy(input.d_data, original.data(), input.sizeBytes(),
                          cudaMemcpyHostToDevice));

    ISPPipeline pipeline;
    pipeline.addBlock(createBlackLevelCorrection(black_level));

    const auto first = download(pipeline.executePreservingInput(input));
    const auto second = download(pipeline.executePreservingInput(input));

    EXPECT_EQ(download(input), original);
    EXPECT_EQ(second, first);
    input.free();
}
