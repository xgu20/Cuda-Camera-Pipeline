#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "blocks.h"
#include "frame_buffer.h"
#include "isp_block.h"

class GammaCorrectionTest : public ::testing::Test {
protected:
  void SetUp() override { CUDA_CHECK(cudaStreamCreate(&stream_)); }

  void TearDown() override { CUDA_CHECK(cudaStreamDestroy(stream_)); }

  cudaStream_t stream_;
};

// ============================================================================
// Correctness Test
// ============================================================================
TEST_F(GammaCorrectionTest, Correctness) {
  const int width = 4;
  const int height = 4;
  const int channels = 3;
  auto gamma_block = createGammaCorrection();
  std::vector<float> h_input(width * height * channels);
  for (int i = 0; i < width * height * channels; ++i) {
    if (i < 4 * channels)
      h_input[i] = .0031308f;
    else if (i < 8 * channels)
      h_input[i] = 0.5f;
    else
      h_input[i] = 1.2f; // Will be clamped to 1.0f -> gamma is 1.0f
  }

  FrameBuffer fb;
  fb.width = width;
  fb.height = height;
  fb.channels = 3;
  fb.format = PixelFormat::RGB_FLOAT;
  fb.allocate();

  // Copy to device
  CUDA_CHECK(cudaMemcpy(fb.d_data, h_input.data(), fb.sizeBytes(),
                        cudaMemcpyHostToDevice));

  // Execute
  FrameBuffer fb_out;
  gamma_block->process(fb, fb_out, stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  // Copy back
  std::vector<float> h_output(width * height * channels);
  CUDA_CHECK(cudaMemcpy(h_output.data(), fb_out.d_data, fb_out.sizeBytes(),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < width * height * channels; ++i) {
    if (i < 4 * channels) {
      EXPECT_FLOAT_EQ(h_output[i], 12.92f * 0.0031308f);
    } else if (i < 8 * channels) {
      EXPECT_NEAR(h_output[i], 0.73535, 0.01f);
    } else {
      EXPECT_NEAR(h_output[i], 1.055f - 0.055f, 0.01f);
    }
  }

  fb.free();
}

// ============================================================================
// Performance Test (Benchmark)
// ============================================================================
TEST_F(GammaCorrectionTest, Performance_4K) {
    const int width = 3840;
    const int height = 2160;
    const int channels = 3;
    const int num_iterations = 100;

    auto gamma_block = createGammaCorrection();

    FrameBuffer fb;
    fb.width = width;
    fb.height = height;
    fb.channels = channels;
    fb.format = PixelFormat::RGB_FLOAT;
    fb.allocate();

    // Initialize with zeros (we don't care about the value for perf testing)
    CUDA_CHECK(cudaMemset(fb.d_data, 0, fb.sizeBytes()));

    // Warm up
    FrameBuffer fb_out;
    gamma_block->process(fb, fb_out, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Setup CUDA Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Run benchmark
    CUDA_CHECK(cudaEventRecord(start, stream_));
    for (int i = 0; i < num_iterations; ++i) {
        gamma_block->process(fb, fb_out, stream_);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    float avg_ms = milliseconds / num_iterations;
    
    // Read + Write bytes per frame
    // Gamma reads 1 float and writes 1 float per pixel per channel
    size_t bytes_per_frame = static_cast<size_t>(width) * height * channels * sizeof(float) * 2;
    float bandwidth_gbps = (bytes_per_frame / (avg_ms * 1e-3f)) / (1024 * 1024 * 1024);

    std::cout << "[ PERF     ] Resolution: " << width << "x" << height << "\n";
    std::cout << "[ PERF     ] Avg Time  : " << avg_ms << " ms\n";
    std::cout << "[ PERF     ] Bandwidth : " << bandwidth_gbps << " GB/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    fb.free();
}

// ============================================================================
// Correctness Test (Optimized)
// ============================================================================
TEST_F(GammaCorrectionTest, Correctness_Optimized) {
  const int width = 4;
  const int height = 4;
  const int channels = 3;
  auto gamma_block = createGammaCorrectionOptimized();
  std::vector<float> h_input(width * height * channels);
  for (int i = 0; i < width * height * channels; ++i) {
    if (i < 4 * channels)
      h_input[i] = .0031308f;
    else if (i < 8 * channels)
      h_input[i] = 0.5f;
    else
      h_input[i] = 1.2f; // Will be clamped to 1.0f -> gamma is 1.0f
  }

  FrameBuffer fb;
  fb.width = width;
  fb.height = height;
  fb.channels = 3;
  fb.format = PixelFormat::RGB_FLOAT;
  fb.allocate();

  // Copy to device
  CUDA_CHECK(cudaMemcpy(fb.d_data, h_input.data(), fb.sizeBytes(),
                        cudaMemcpyHostToDevice));

  // Execute
  FrameBuffer fb_out;
  gamma_block->process(fb, fb_out, stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  // Copy back
  std::vector<float> h_output(width * height * channels);
  CUDA_CHECK(cudaMemcpy(h_output.data(), fb_out.d_data, fb_out.sizeBytes(),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < width * height * channels; ++i) {
    if (i < 4 * channels) {
      EXPECT_FLOAT_EQ(h_output[i], 12.92f * 0.0031308f);
    } else if (i < 8 * channels) {
      EXPECT_NEAR(h_output[i], 0.73535, 0.01f);
    } else {
      EXPECT_NEAR(h_output[i], 1.055f - 0.055f, 0.01f);
    }
  }

  fb.free();
}

// ============================================================================
// Performance Test (Benchmark - Optimized)
// ============================================================================
TEST_F(GammaCorrectionTest, Performance_4K_Optimized) {
    const int width = 3840;
    const int height = 2160;
    const int channels = 3;
    const int num_iterations = 100;

    auto gamma_block = createGammaCorrectionOptimized();

    FrameBuffer fb;
    fb.width = width;
    fb.height = height;
    fb.channels = channels;
    fb.format = PixelFormat::RGB_FLOAT;
    fb.allocate();

    // Initialize with zeros
    CUDA_CHECK(cudaMemset(fb.d_data, 0, fb.sizeBytes()));

    // Warm up
    FrameBuffer fb_out;
    gamma_block->process(fb, fb_out, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Setup CUDA Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Run benchmark
    CUDA_CHECK(cudaEventRecord(start, stream_));
    for (int i = 0; i < num_iterations; ++i) {
        gamma_block->process(fb, fb_out, stream_);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    float avg_ms = milliseconds / num_iterations;
    
    size_t bytes_per_frame = static_cast<size_t>(width) * height * channels * sizeof(float) * 2;
    float bandwidth_gbps = (bytes_per_frame / (avg_ms * 1e-3f)) / (1024 * 1024 * 1024);

    std::cout << "[ PERF     ] Resolution: " << width << "x" << height << "\n";
    std::cout << "[ PERF     ] Avg Time  : " << avg_ms << " ms\n";
    std::cout << "[ PERF     ] Bandwidth : " << bandwidth_gbps << " GB/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    fb.free();
}