#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <iostream>
#include <iomanip>

#include "blocks.h"
#include "frame_buffer.h"
#include "isp_block.h"

class LensShadingCorrectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    void TearDown() override {
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }

    cudaStream_t stream_;
};

// Expected gain using bilinear interpolation formula with channel multipliers
inline float getExpectedGainChannel(int x, int y, PixelFormat format) {
    float u = static_cast<float>(x) / 3.0f;
    float v = static_cast<float>(y) / 3.0f;
    float base_gain = (1.0f - u) * (1.0f - v) * 1.0f +
                      u * (1.0f - v) * 2.0f +
                      (1.0f - u) * v * 3.0f +
                      u * v * 4.0f;

    // Determine the channel multiplier based on Bayer phase
    // lut[0] = R  (mult = 1.0f)
    // lut[1] = Gr (mult = 1.1f)
    // lut[2] = Gb (mult = 1.2f)
    // lut[3] = B  (mult = 1.3f)
    int rx = 0, ry = 0;
    if (format == PixelFormat::BAYER_RGGB) {
        rx = 0; ry = 0;
    } else if (format == PixelFormat::BAYER_BGGR) {
        rx = 1; ry = 1;
    } else if (format == PixelFormat::BAYER_GRBG) {
        rx = 1; ry = 0;
    } else if (format == PixelFormat::BAYER_GBRG) {
        rx = 0; ry = 1;
    }

    bool is_red = (x & 1) == rx && (y & 1) == ry;
    bool is_blue = (x & 1) == (1 - rx) && (y & 1) == (1 - ry);
    bool is_gr = (x & 1) == (1 - rx) && (y & 1) == ry;

    float mult = 1.0f;
    if (is_red) {
        mult = 1.0f;
    } else if (is_gr) {
        mult = 1.1f;
    } else if (is_blue) {
        mult = 1.3f;
    } else {
        mult = 1.2f;
    }

    return base_gain * mult;
}

// ============================================================================
// Correctness Test
// ============================================================================
TEST_F(LensShadingCorrectionTest, Correctness) {
    const int width = 4;
    const int height = 4;
    const int grid_w = 2;
    const int grid_h = 2;
    const int bit_depth = 12;

    std::vector<float> channel_lut = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<std::vector<float>> lut(4);
    std::vector<float> multipliers = { 1.0f, 1.1f, 1.2f, 1.3f };
    for (int c = 0; c < 4; ++c) {
        lut[c].resize(4);
        for (int i = 0; i < 4; ++i) {
            lut[c][i] = channel_lut[i] * multipliers[c];
        }
    }

    auto lsc_block = createLensShadingCorrection(lut, grid_w, grid_h, bit_depth);

    std::vector<uint16_t> h_input(width * height, 100);

    FrameBuffer fb;
    fb.width = width;
    fb.height = height;
    fb.channels = 1;
    fb.format = PixelFormat::BAYER_RGGB;
    fb.allocate();

    CUDA_CHECK(cudaMemcpy(fb.d_data, h_input.data(), fb.sizeBytes(), cudaMemcpyHostToDevice));

    FrameBuffer fb_out;
    lsc_block->process(fb, fb_out, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<uint16_t> h_output(width * height);
    CUDA_CHECK(cudaMemcpy(h_output.data(), fb.d_data, fb.sizeBytes(), cudaMemcpyDeviceToHost));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float gain = getExpectedGainChannel(x, y, PixelFormat::BAYER_RGGB);
            float expected_val = 100.0f * gain;
            uint16_t actual_val = h_output[y * width + x];
            EXPECT_NEAR(actual_val, expected_val, 1.5f) 
                << "Mismatch at pixel (" << x << ", " << y << ")";
        }
    }

    fb.free();
}

// ============================================================================
// Performance Test
// ============================================================================
TEST_F(LensShadingCorrectionTest, Performance_4K) {
    const int width = 3840;
    const int height = 2160;
    const int grid_w = 33;
    const int grid_h = 33;
    const int bit_depth = 12;
    const int num_iterations = 100;
    const int num_buffers = 10;

    std::vector<float> channel_lut(grid_w * grid_h, 1.2f);
    std::vector<std::vector<float>> lut(4, channel_lut);

    auto block_lsc = createLensShadingCorrection(lut, grid_w, grid_h, bit_depth);

    std::vector<FrameBuffer> fb_array(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
        fb_array[i].width = width;
        fb_array[i].height = height;
        fb_array[i].channels = 1;
        fb_array[i].format = PixelFormat::BAYER_RGGB;
        fb_array[i].allocate();
        CUDA_CHECK(cudaMemset(fb_array[i].d_data, 0, fb_array[i].sizeBytes()));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    FrameBuffer fb_out;

    // Warm-up
    block_lsc->process(fb_array[0], fb_out, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    CUDA_CHECK(cudaEventRecord(start, stream_));
    for (int i = 0; i < num_iterations; ++i) {
        block_lsc->process(fb_array[i % num_buffers], fb_out, stream_);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_lsc = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_lsc, start, stop));
    float avg_ms_lsc = ms_lsc / num_iterations;

    size_t bytes_per_frame = static_cast<size_t>(width) * height * sizeof(uint16_t) * 2;
    float bandwidth = (bytes_per_frame / (avg_ms_lsc * 1e-3f)) / (1024.0f * 1024.0f * 1024.0f);

    std::cout << "\n============================================================\n";
    std::cout << " LSC Performance Benchmark (4K Resolution: 3840x2160)\n";
    std::cout << "============================================================\n";
    std::cout << "Avg Time:  " << avg_ms_lsc << " ms\n";
    std::cout << "Bandwidth: " << bandwidth << " GB/s\n";
    std::cout << "============================================================\n\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    for (int i = 0; i < num_buffers; ++i) {
        fb_array[i].free();
    }
}
