#include "blocks.h"
#include "frame_buffer.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

namespace {
void runDenoise(const std::vector<float>& h_in, int w, int h,
                const YuvDenoiseConfig& config, std::vector<float>& h_out) {

    FrameBuffer input;
    input.width = w;
    input.height = h;
    input.channels = 3;
    input.format = PixelFormat::YUV_FLOAT;
    input.allocate();

    CUDA_CHECK(cudaMemcpy(input.d_data, h_in.data(), h_in.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    FrameBuffer output;
    
    auto denoise = createYuvDenoise(config);
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    denoise->process(input, output, stream);
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

TEST(YuvDenoiseTest, ConstantYuvIsPreservedAcrossPartialTiles) {
    constexpr int w = 17;
    constexpr int h = 19; // exercise partial tiles on both axes
    std::vector<float> input(w * h * 3);
    for (int i = 0; i < w * h; ++i) {
        input[i * 3] = 0.8f;
        input[i * 3 + 1] = 0.3f;
        input[i * 3 + 2] = 0.1f;
    }

    std::vector<float> output;
    runDenoise(input, w, h, YuvDenoiseConfig{}, output);
    for (int i = 0; i < w * h; ++i) {
        EXPECT_NEAR(output[i * 3], 0.8f, 2e-5f) << "pixel " << i;
        EXPECT_NEAR(output[i * 3 + 1], 0.3f, 2e-5f) << "pixel " << i;
        EXPECT_NEAR(output[i * 3 + 2], 0.1f, 2e-5f) << "pixel " << i;
    }
}

TEST(YuvDenoiseTest, StrengthControlsLumaAndChromaFiltering) {
    constexpr int w = 17, h = 17, cx = 8, cy = 8;
    std::vector<float> input(w * h * 3, 0.5f);
    input[(cy * w + cx) * 3] = 0.6f;
    input[(cy * w + cx) * 3 + 2] = 1.0f;

    YuvDenoiseConfig bypass;
    bypass.luma_strength = 0.0f;
    bypass.chroma_strength = 0.0f;
    std::vector<float> unchanged;
    runDenoise(input, w, h, bypass, unchanged);

    YuvDenoiseConfig filtered;
    filtered.luma_strength = 1.0f;
    filtered.chroma_strength = 1.0f;
    filtered.luma_range_sigma = 1.0f;
    filtered.chroma_range_sigma = 1.0f;
    std::vector<float> smoothed;
    runDenoise(input, w, h, filtered, smoothed);

    const int p = (cy * w + cx) * 3;
    EXPECT_NEAR(unchanged[p], 0.6f, 2e-5f);
    EXPECT_NEAR(unchanged[p + 2], 1.0f, 2e-5f);
    EXPECT_LT(smoothed[p], unchanged[p]);
    EXPECT_LT(smoothed[p + 2], unchanged[p + 2]);
}

TEST(YuvDenoiseTest, RejectsInvalidConfigurationAndFormat) {
    YuvDenoiseConfig bad;
    bad.spatial_sigma = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(createYuvDenoise(bad), std::invalid_argument);
    bad = YuvDenoiseConfig{};
    bad.luma_strength = 1.1f;
    EXPECT_THROW(createYuvDenoise(bad), std::invalid_argument);

    FrameBuffer wrong;
    wrong.width = wrong.height = 1;
    wrong.channels = 3;
    wrong.format = PixelFormat::RGB_FLOAT;
    auto block = createYuvDenoise(YuvDenoiseConfig{});
    FrameBuffer output;
    EXPECT_THROW(block->process(wrong, output, nullptr), std::invalid_argument);
}

// ============================================================================
// Performance Test
// ============================================================================
TEST(YuvDenoiseTest, Performance_4K) {
    const int width = 3840;
    const int height = 2160;
    const int num_iterations = 100;
    const int num_buffers = 2;

    YuvDenoiseConfig config;
    auto denoise = createYuvDenoise(config);

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
    denoise->process(fb_array[0], fb_out, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < num_iterations; ++i) {
        denoise->process(fb_array[i % num_buffers], fb_out, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / num_iterations;

    // Read 3 floats, write 3 floats per pixel
    size_t bytes_per_frame = static_cast<size_t>(width) * height * sizeof(float) * 6;
    float bandwidth = (bytes_per_frame / (avg_ms * 1e-3f)) / (1024.0f * 1024.0f * 1024.0f);

    std::cout << "\n============================================================\n";
    std::cout << " YUV Denoise Performance Benchmark (4K Resolution: 3840x2160)\n";
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
