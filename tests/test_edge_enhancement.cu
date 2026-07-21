#include "blocks.h"
#include "frame_buffer.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

namespace {
void runEdge(const std::vector<float>& h_in, int w, int h,
             const EdgeEnhancementConfig& config, std::vector<float>& h_out) {

    FrameBuffer input;
    input.width = w;
    input.height = h;
    input.channels = 3;
    input.format = PixelFormat::YUV_FLOAT;
    input.allocate();

    CUDA_CHECK(cudaMemcpy(input.d_data, h_in.data(), h_in.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    FrameBuffer output;
    auto block = createEdgeEnhancement(config);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    block->process(input, output, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(output.width, w);
    EXPECT_EQ(output.height, h);
    EXPECT_EQ(output.channels, 3);
    EXPECT_EQ(output.format, PixelFormat::YUV_FLOAT);
    
    h_out.resize(h_in.size());
    CUDA_CHECK(cudaMemcpy(h_out.data(), output.d_data,
                          h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    input.free();
    output.free();
    CUDA_CHECK(cudaStreamDestroy(stream));
}
} // namespace

TEST(EdgeEnhancementTest, StepEdgeMatchesGoldenResponse) {
    constexpr int w = 17, h = 19; // partial tiles
    std::vector<float> input(w * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int p = (y * w + x) * 3;
            input[p] = x < 8 ? 0.5f : 0.8f;
            input[p + 1] = 0.6f;
            input[p + 2] = 0.4f;
        }
    }
    EdgeEnhancementConfig config;
    config.strength = 1.5f;
    config.threshold = 0.01f;
    config.clamp_limit = 0.2f;
    std::vector<float> output;
    runEdge(input, w, h, config, output);

    const int row = 9;
    // 5x5 box blur gives 0.62 at x=7 and 0.68 at x=8. After soft
    // thresholding and strength, enhanced Y values are 0.335 and 0.965.
    EXPECT_NEAR(output[(row * w + 7) * 3], 0.335f, 2e-5f);
    EXPECT_NEAR(output[(row * w + 8) * 3], 0.965f, 2e-5f);

    // A non-neutral flat area verifies that output channels are RGB, not U/V.
    const int flat = (row * w + 3) * 3;
    EXPECT_NEAR(output[flat], 0.5f, 2e-5f);
    EXPECT_NEAR(output[flat + 1], 0.6f, 2e-5f);
    EXPECT_NEAR(output[flat + 2], 0.4f, 2e-5f);
}

TEST(EdgeEnhancementTest, RejectsInvalidConfigurationAndFormat) {
    EdgeEnhancementConfig bad;
    bad.threshold = -0.01f;
    EXPECT_THROW(createEdgeEnhancement(bad), std::invalid_argument);
    bad = EdgeEnhancementConfig{};
    bad.strength = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(createEdgeEnhancement(bad), std::invalid_argument);

    FrameBuffer wrong;
    wrong.width = wrong.height = 1;
    wrong.channels = 3;
    wrong.format = PixelFormat::RGB_FLOAT;
    auto block = createEdgeEnhancement(EdgeEnhancementConfig{});
    FrameBuffer output;
    EXPECT_THROW(block->process(wrong, output, nullptr), std::invalid_argument);
}

// ============================================================================
// Performance Test
// ============================================================================
TEST(EdgeEnhancementTest, Performance_4K) {
    const int width = 3840;
    const int height = 2160;
    const int num_iterations = 100;
    const int num_buffers = 2;

    EdgeEnhancementConfig config;
    auto block = createEdgeEnhancement(config);

    std::vector<FrameBuffer> fb_array(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
        fb_array[i].width = width;
        fb_array[i].height = height;
        fb_array[i].channels = 3;
        fb_array[i].format = PixelFormat::YUV_FLOAT;
        fb_array[i].allocate();
        CUDA_CHECK(cudaMemset(fb_array[i].d_data, 0, fb_array[i].sizeBytes()));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    FrameBuffer fb_out;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm-up
    block->process(fb_array[0], fb_out, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < num_iterations; ++i) {
        block->process(fb_array[i % num_buffers], fb_out, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / num_iterations;

    size_t bytes_per_frame = static_cast<size_t>(width) * height * sizeof(float) * 6;
    float bandwidth = (bytes_per_frame / (avg_ms * 1e-3f)) / (1024.0f * 1024.0f * 1024.0f);

    std::cout << "\n============================================================\n";
    std::cout << " Edge Enhancement Performance Benchmark (4K Resolution: 3840x2160)\n";
    std::cout << "============================================================\n";
    std::cout << "Avg Time:  " << avg_ms << " ms\n";
    std::cout << "Bandwidth: " << bandwidth << " GB/s\n";
    std::cout << "============================================================\n\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    for (int i = 0; i < num_buffers; ++i) {
        fb_array[i].free();
    }
    fb_out.free();
}
