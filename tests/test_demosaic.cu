#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
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

using DemosaicFactory = std::unique_ptr<ISPBlock>(*)(int);

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
// Per-pattern dispatch + correctness — parameterised over all 4 Bayer
// patterns × {Naive, Optimized}. Ground-truth check (not just naive-vs-opt
// agreement) catches dispatch-table bugs that would otherwise look fine.
//
// Construction trick: filling R-positions with a single value, B-positions
// with another, G-positions with a third produces a Bayer plane whose
// 4-cardinal and 4-diagonal neighbours of any pixel are all the same value,
// so bilinear demosaic at interior pixels reproduces (R, G, B) exactly.
// ============================================================================
struct BayerPatternCase {
    PixelFormat fmt;
    int         red_bx;  // x position of R within the 2x2 cell
    int         red_by;  // y position of R within the 2x2 cell
    const char* name;
};

class DemosaicPatternTest : public ::testing::TestWithParam<BayerPatternCase> {
protected:
    void SetUp() override { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    void TearDown() override { CUDA_CHECK(cudaStreamDestroy(stream_)); }
    cudaStream_t stream_;
};

TEST_P(DemosaicPatternTest, DispatchesToCorrectSpecialization) {
    const auto& tc = GetParam();
    const int width = 64, height = 64, bit_depth = 10;
    const float max_val = static_cast<float>((1 << bit_depth) - 1);
    const uint16_t R_VAL = 800, G_VAL = 500, B_VAL = 200;

    auto pixel_kind = [&](int x, int y) -> char {
        const int bx = x & 1, by = y & 1;
        const int blue_bx = 1 - tc.red_bx, blue_by = 1 - tc.red_by;
        if (bx == tc.red_bx && by == tc.red_by) return 'R';
        if (bx == blue_bx  && by == blue_by)   return 'B';
        return 'G';
    };

    std::vector<uint16_t> h_in(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const char k = pixel_kind(x, y);
            h_in[y * width + x] = (k == 'R') ? R_VAL : (k == 'B') ? B_VAL : G_VAL;
        }
    }

    FrameBuffer fb_in;
    fb_in.width    = width;
    fb_in.height   = height;
    fb_in.channels = 1;
    fb_in.format   = tc.fmt;
    fb_in.allocate();
    CUDA_CHECK(cudaMemcpy(fb_in.d_data, h_in.data(), fb_in.sizeBytes(),
                          cudaMemcpyHostToDevice));

    const struct { const char* name; DemosaicFactory factory; } variants[] = {
        {"Naive",     &createDemosaic},
        {"Optimized", &createDemosaicOptimized},
    };

    const float exp_r = R_VAL / max_val;
    const float exp_g = G_VAL / max_val;
    const float exp_b = B_VAL / max_val;

    for (const auto& v : variants) {
        auto block = v.factory(bit_depth);
        FrameBuffer fb_out;
        block->process(fb_in, fb_out, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        std::vector<float> h_out(width * height * 3);
        CUDA_CHECK(cudaMemcpy(h_out.data(), fb_out.d_data, fb_out.sizeBytes(),
                              cudaMemcpyDeviceToHost));

        // Skip 2-pixel border so the boundary clamp/halo doesn't matter.
        for (int y = 2; y < height - 2; ++y) {
            for (int x = 2; x < width - 2; ++x) {
                const int idx = (y * width + x) * 3;
                EXPECT_NEAR(h_out[idx + 0], exp_r, 1e-4f)
                    << v.name << " R mismatch at (" << x << "," << y << ")";
                EXPECT_NEAR(h_out[idx + 1], exp_g, 1e-4f)
                    << v.name << " G mismatch at (" << x << "," << y << ")";
                EXPECT_NEAR(h_out[idx + 2], exp_b, 1e-4f)
                    << v.name << " B mismatch at (" << x << "," << y << ")";
            }
        }

        fb_out.free();
    }

    fb_in.free();
}

INSTANTIATE_TEST_SUITE_P(
    AllPatterns,
    DemosaicPatternTest,
    ::testing::Values(
        BayerPatternCase{PixelFormat::BAYER_RGGB, 0, 0, "RGGB"},
        BayerPatternCase{PixelFormat::BAYER_BGGR, 1, 1, "BGGR"},
        BayerPatternCase{PixelFormat::BAYER_GRBG, 1, 0, "GRBG"},
        BayerPatternCase{PixelFormat::BAYER_GBRG, 0, 1, "GBRG"}),
    [](const ::testing::TestParamInfo<BayerPatternCase>& info) {
        return std::string(info.param.name);
    });

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
