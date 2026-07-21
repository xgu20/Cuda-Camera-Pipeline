#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "blocks.h"
#include "frame_buffer.h"

TEST(AutoWhiteBalanceTest, ManualGainClampsAtFullBitDepthRange) {
    constexpr int width = 4;
    constexpr int height = 4;
    constexpr int bit_depth = 10;

    std::vector<uint16_t> input(width * height, 800);

    FrameBuffer frame;
    frame.width = width;
    frame.height = height;
    frame.channels = 1;
    frame.format = PixelFormat::BAYER_RGGB;
    frame.packing = PixelPacking::UNPACKED_U16;
    frame.bit_depth = bit_depth;
    frame.allocate();
    CUDA_CHECK(cudaMemcpy(frame.d_data, input.data(), frame.sizeBytes(),
                          cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto awb = createManualWhiteBalance({2.0f, 2.0f, 2.0f, 2.0f}, bit_depth);
    FrameBuffer output;
    awb->process(frame, output, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<uint16_t> result(width * height);
    CUDA_CHECK(cudaMemcpy(result.data(), output.d_data, output.sizeBytes(),
                          cudaMemcpyDeviceToHost));
    for (uint16_t pixel : result) {
        EXPECT_EQ(pixel, 1023);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    frame.free();
}
