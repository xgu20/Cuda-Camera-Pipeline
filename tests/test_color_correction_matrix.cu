#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "blocks.h"
#include "frame_buffer.h"

TEST(ColorCorrectionMatrixTest, AppliesThreeByThreeMatrix) {
    const std::vector<float> input = {1.0f, 2.0f, 3.0f};
    ColorCorrectionMatrix matrix{{
        1.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 1.0f,
        1.0f, 0.0f, 1.0f,
    }};

    FrameBuffer frame;
    frame.width = 1;
    frame.height = 1;
    frame.channels = 3;
    frame.format = PixelFormat::RGB_FLOAT;
    frame.allocate();
    CUDA_CHECK(cudaMemcpy(frame.d_data, input.data(), frame.sizeBytes(),
                          cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto ccm = createColorCorrectionMatrix(matrix);
    FrameBuffer output;
    ccm->process(frame, output, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> result(3);
    CUDA_CHECK(cudaMemcpy(result.data(), output.d_data, output.sizeBytes(),
                          cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(result[0], 3.0f);
    EXPECT_FLOAT_EQ(result[1], 5.0f);
    EXPECT_FLOAT_EQ(result[2], 4.0f);

    CUDA_CHECK(cudaStreamDestroy(stream));
    frame.free();
}
