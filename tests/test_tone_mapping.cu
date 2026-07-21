#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>

#include "blocks.h"
#include "frame_buffer.h"

class ToneMappingTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    void TearDown() override {
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }

    cudaStream_t stream_;
};

TEST_F(ToneMappingTest, BasicFunctionality) {
    const int width = 4;
    const int height = 4;
    const float exposure = 2.0f;

    auto tm_block = createToneMapping(exposure);

    std::vector<float> h_input(width * height * 3, 0.5f);
    h_input[0] = 0.0f; // Black
    h_input[1] = 0.5f; // Mid
    h_input[2] = 2.0f; // Highlight
    
    FrameBuffer fb;
    fb.width = width;
    fb.height = height;
    fb.channels = 3;
    fb.format = PixelFormat::RGB_FLOAT;
    fb.allocate();

    CUDA_CHECK(cudaMemcpy(fb.d_data, h_input.data(), fb.sizeBytes(), cudaMemcpyHostToDevice));

    FrameBuffer fb_out;
    tm_block->process(fb, fb_out, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<float> h_output(width * height * 3);
    CUDA_CHECK(cudaMemcpy(h_output.data(), fb.d_data, fb.sizeBytes(), cudaMemcpyDeviceToHost));

    // Verify it's clamped to [0, 1]
    for (size_t i = 0; i < h_output.size(); ++i) {
        EXPECT_GE(h_output[i], 0.0f);
        EXPECT_LE(h_output[i], 1.0f);
    }
    
    // Black should stay 0
    EXPECT_NEAR(h_output[0], 0.0f, 1e-5f);

    fb.free();
}

TEST_F(ToneMappingTest, Performance_4K) {
    const int width = 3840;
    const int height = 2160;
    const int num_iterations = 100;
    const int num_buffers = 10;

    auto tm_block = createToneMapping(1.0f);

    std::vector<FrameBuffer> fb_array(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
        fb_array[i].width = width;
        fb_array[i].height = height;
        fb_array[i].channels = 3;
        fb_array[i].format = PixelFormat::RGB_FLOAT;
        fb_array[i].allocate();
        CUDA_CHECK(cudaMemset(fb_array[i].d_data, 0, fb_array[i].sizeBytes()));
    }

    FrameBuffer fb_out;
    tm_block->process(fb_array[0], fb_out, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream_));
    for (int i = 0; i < num_iterations; ++i) {
        tm_block->process(fb_array[i % num_buffers], fb_out, stream_);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_ms = milliseconds / num_iterations;
    
    size_t bytes_per_frame = static_cast<size_t>(width) * height * 3 * sizeof(float) * 2;
    float bandwidth_gbps = (bytes_per_frame / (avg_ms * 1e-3f)) / (1024 * 1024 * 1024);

    std::cout << "\n============================================================\n";
    std::cout << " Tone Mapping Performance Benchmark (4K Resolution: 3840x2160)\n";
    std::cout << "============================================================\n";
    std::cout << "Avg Time:  " << avg_ms << " ms\n";
    std::cout << "Bandwidth: " << bandwidth_gbps << " GB/s\n";
    std::cout << "============================================================\n\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    for (int i = 0; i < num_buffers; ++i) {
        fb_array[i].free();
    }
}
