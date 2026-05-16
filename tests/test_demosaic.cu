#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <iostream>

#include "blocks.h"
#include "frame_buffer.h"
#include "isp_block.h"

class DemosaicTest : public ::testing::Test {
protected:
    void SetUp() override {
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
TEST_F(DemosaicTest, CorrectnessMatch) {
    const int width = 128;
    const int height = 128;
    const int bit_depth = 10;
    
    auto demosaic_naive = createDemosaic(bit_depth);
    auto demosaic_opt = createDemosaicOptimized(bit_depth);

    std::vector<uint16_t> h_input(width * height);
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = rand() % 1024; // 10-bit random values
    }

    FrameBuffer fb_in;
    fb_in.width = width;
    fb_in.height = height;
    fb_in.channels = 1;
    fb_in.format = PixelFormat::BAYER_RGGB;
    fb_in.allocate();
    CUDA_CHECK(cudaMemcpy(fb_in.d_data, h_input.data(), fb_in.sizeBytes(), cudaMemcpyHostToDevice));

    FrameBuffer fb_out_naive;
    demosaic_naive->process(fb_in, fb_out_naive, stream_);
    
    FrameBuffer fb_out_opt;
    demosaic_opt->process(fb_in, fb_out_opt, stream_);
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<float> h_out_naive(width * height * 3);
    std::vector<float> h_out_opt(width * height * 3);
    
    CUDA_CHECK(cudaMemcpy(h_out_naive.data(), fb_out_naive.d_data, fb_out_naive.sizeBytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_opt.data(), fb_out_opt.d_data, fb_out_opt.sizeBytes(), cudaMemcpyDeviceToHost));

    // Verify
    for (int i = 0; i < width * height * 3; ++i) {
        EXPECT_NEAR(h_out_naive[i], h_out_opt[i], 1e-5f) << "Mismatch at index " << i;
    }

    fb_in.free();
    fb_out_naive.free();
    fb_out_opt.free();
}

// ============================================================================
// Performance Test (Benchmark)
// ============================================================================
TEST_F(DemosaicTest, Performance_4K) {
    const int width = 3840;
    const int height = 2160;
    const int bit_depth = 10;
    const int num_iterations = 100;

    auto demosaic_naive = createDemosaic(bit_depth);
    auto demosaic_opt = createDemosaicOptimized(bit_depth);

    FrameBuffer fb_in;
    fb_in.width = width;
    fb_in.height = height;
    fb_in.channels = 1;
    fb_in.format = PixelFormat::BAYER_RGGB;
    fb_in.allocate();
    CUDA_CHECK(cudaMemset(fb_in.d_data, 0, fb_in.sizeBytes()));

    FrameBuffer fb_out_naive, fb_out_opt;
    demosaic_naive->process(fb_in, fb_out_naive, stream_);
    demosaic_opt->process(fb_in, fb_out_opt, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Benchmark Naive
    CUDA_CHECK(cudaEventRecord(start, stream_));
    for (int i = 0; i < num_iterations; ++i) {
        demosaic_naive->process(fb_in, fb_out_naive, stream_);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_naive = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
    ms_naive /= num_iterations;

    // Benchmark Optimized
    CUDA_CHECK(cudaEventRecord(start, stream_));
    for (int i = 0; i < num_iterations; ++i) {
        demosaic_opt->process(fb_in, fb_out_opt, stream_);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_opt = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_opt, start, stop));
    ms_opt /= num_iterations;
    
    // Read 1 uint16_t, Write 3 floats per pixel
    size_t bytes_per_frame = static_cast<size_t>(width) * height * (sizeof(uint16_t) + 3 * sizeof(float));
    float bw_naive = (bytes_per_frame / (ms_naive * 1e-3f)) / (1024 * 1024 * 1024);
    float bw_opt = (bytes_per_frame / (ms_opt * 1e-3f)) / (1024 * 1024 * 1024);

    std::cout << "[ PERF NAIVE ] Avg Time  : " << ms_naive << " ms (" << bw_naive << " GB/s)\n";
    std::cout << "[ PERF OPTIM ] Avg Time  : " << ms_opt << " ms (" << bw_opt << " GB/s)\n";
    std::cout << "[ SPEEDUP    ] " << ms_naive / ms_opt << "x\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    fb_in.free();
    fb_out_naive.free();
    fb_out_opt.free();
}
