#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

#include "blocks.h"
#include "frame_buffer.h"
#include "isp_block.h"

class BlackLevelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create CUDA stream for the tests
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    void TearDown() override {
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }

    cudaStream_t stream_;
};

// ============================================================================
// Correctness Test
// ============================================================================
TEST_F(BlackLevelTest, Correctness) {
    const int width = 4;
    const int height = 4;
    const uint16_t black_level = 64;

    // Create block
    auto blc_block = createBlackLevelCorrection(black_level);

    // Prepare host data
    std::vector<uint16_t> h_input(width * height);
    for (int i = 0; i < width * height; ++i) {
        // Create some values: below black level, at black level, above black level
        if (i < 4) h_input[i] = 10;            // Should clamp to 0
        else if (i < 8) h_input[i] = 64;       // Should become 0
        else h_input[i] = 100 + i;             // Should become 100+i - 64
    }

    // Prepare device buffer
    FrameBuffer fb;
    fb.width = width;
    fb.height = height;
    fb.channels = 1;
    fb.format = PixelFormat::BAYER_RGGB;
    fb.allocate();

    // Copy to device
    CUDA_CHECK(cudaMemcpy(fb.d_data, h_input.data(), fb.sizeBytes(), cudaMemcpyHostToDevice));

    // Execute
    FrameBuffer fb_out; // Will alias fb in-place
    blc_block->process(fb, fb_out, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Copy back
    std::vector<uint16_t> h_output(width * height);
    CUDA_CHECK(cudaMemcpy(h_output.data(), fb.d_data, fb.sizeBytes(), cudaMemcpyDeviceToHost));

    // Verify
    for (int i = 0; i < width * height; ++i) {
        if (i < 4) {
            EXPECT_EQ(h_output[i], 0) << "Failed clamping at index " << i;
        } else if (i < 8) {
            EXPECT_EQ(h_output[i], 0) << "Failed exact black level at index " << i;
        } else {
            EXPECT_EQ(h_output[i], 100 + i - 64) << "Failed subtraction at index " << i;
        }
    }

    fb.free();
}

// ============================================================================
// Performance Test (Benchmark)
// ============================================================================
TEST_F(BlackLevelTest, Performance_4K) {
    const int width = 3840;
    const int height = 2160;
    const uint16_t black_level = 64;
    const int num_iterations = 100;
    const int num_buffers = 10;

    auto blc_block = createBlackLevelCorrection(black_level);

    std::vector<FrameBuffer> fb_array(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
        fb_array[i].width = width;
        fb_array[i].height = height;
        fb_array[i].channels = 1;
        fb_array[i].format = PixelFormat::BAYER_RGGB;
        fb_array[i].allocate();
        CUDA_CHECK(cudaMemset(fb_array[i].d_data, 0, fb_array[i].sizeBytes()));
    }

    // Warm up
    FrameBuffer fb_out;
    blc_block->process(fb_array[0], fb_out, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Setup CUDA Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Run benchmark
    CUDA_CHECK(cudaEventRecord(start, stream_));
    for (int i = 0; i < num_iterations; ++i) {
        blc_block->process(fb_array[i % num_buffers], fb_out, stream_);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    float avg_ms = milliseconds / num_iterations;
    
    // Read + Write bytes per frame
    // BLC reads 1 uint16_t and writes 1 uint16_t per pixel
    size_t bytes_per_frame = static_cast<size_t>(width) * height * sizeof(uint16_t) * 2;
    float bandwidth_gbps = (bytes_per_frame / (avg_ms * 1e-3f)) / (1024 * 1024 * 1024);

    std::cout << "[ PERF     ] Resolution: " << width << "x" << height << "\n";
    std::cout << "[ PERF     ] Avg Time  : " << avg_ms << " ms\n";
    std::cout << "[ PERF     ] Bandwidth : " << bandwidth_gbps << " GB/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    for (int i = 0; i < num_buffers; ++i) {
        fb_array[i].free();
    }
}

// ============================================================================
// Correctness Test (Optimized)
// ============================================================================
TEST_F(BlackLevelTest, Correctness_Optimized) {
    const int width = 4;
    const int height = 4;
    const uint16_t black_level = 64;

    // Create block
    auto blc_block = createBlackLevelCorrectionOptimized(black_level);

    // Prepare host data
    std::vector<uint16_t> h_input(width * height);
    for (int i = 0; i < width * height; ++i) {
        // Create some values: below black level, at black level, above black level
        if (i < 4) h_input[i] = 10;            // Should clamp to 0
        else if (i < 8) h_input[i] = 64;       // Should become 0
        else h_input[i] = 100 + i;             // Should become 100+i - 64
    }

    // Prepare device buffer
    FrameBuffer fb;
    fb.width = width;
    fb.height = height;
    fb.channels = 1;
    fb.format = PixelFormat::BAYER_RGGB;
    fb.allocate();

    // Copy to device
    CUDA_CHECK(cudaMemcpy(fb.d_data, h_input.data(), fb.sizeBytes(), cudaMemcpyHostToDevice));

    // Execute
    FrameBuffer fb_out; // Will alias fb in-place
    blc_block->process(fb, fb_out, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Copy back
    std::vector<uint16_t> h_output(width * height);
    CUDA_CHECK(cudaMemcpy(h_output.data(), fb.d_data, fb.sizeBytes(), cudaMemcpyDeviceToHost));

    // Verify
    for (int i = 0; i < width * height; ++i) {
        if (i < 4) {
            EXPECT_EQ(h_output[i], 0) << "Failed clamping at index " << i;
        } else if (i < 8) {
            EXPECT_EQ(h_output[i], 0) << "Failed exact black level at index " << i;
        } else {
            EXPECT_EQ(h_output[i], 100 + i - 64) << "Failed subtraction at index " << i;
        }
    }

    fb.free();
}

// ============================================================================
// Performance Test (Benchmark - Optimized)
// ============================================================================
TEST_F(BlackLevelTest, Performance_4K_Optimized) {
    const int width = 3840;
    const int height = 2160;
    const uint16_t black_level = 64;
    const int num_iterations = 100;
    const int num_buffers = 10;

    auto blc_block = createBlackLevelCorrectionOptimized(black_level);

    std::vector<FrameBuffer> fb_array(num_buffers);
    for (int i = 0; i < num_buffers; ++i) {
        fb_array[i].width = width;
        fb_array[i].height = height;
        fb_array[i].channels = 1;
        fb_array[i].format = PixelFormat::BAYER_RGGB;
        fb_array[i].allocate();
        CUDA_CHECK(cudaMemset(fb_array[i].d_data, 0, fb_array[i].sizeBytes()));
    }

    // Warm up
    FrameBuffer fb_out;
    blc_block->process(fb_array[0], fb_out, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Setup CUDA Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Run benchmark
    CUDA_CHECK(cudaEventRecord(start, stream_));
    for (int i = 0; i < num_iterations; ++i) {
        blc_block->process(fb_array[i % num_buffers], fb_out, stream_);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    float avg_ms = milliseconds / num_iterations;
    
    // Read + Write bytes per frame
    size_t bytes_per_frame = static_cast<size_t>(width) * height * sizeof(uint16_t) * 2;
    float bandwidth_gbps = (bytes_per_frame / (avg_ms * 1e-3f)) / (1024 * 1024 * 1024);

    std::cout << "[ PERF     ] Resolution: " << width << "x" << height << "\n";
    std::cout << "[ PERF     ] Avg Time  : " << avg_ms << " ms\n";
    std::cout << "[ PERF     ] Bandwidth : " << bandwidth_gbps << " GB/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    for (int i = 0; i < num_buffers; ++i) {
        fb_array[i].free();
    }
}
