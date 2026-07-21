#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "blocks.h"
#include "frame_buffer.h"
#include "isp_pipeline.h"

namespace {

std::vector<uint16_t> download(const FrameBuffer& frame) {
    std::vector<uint16_t> host(static_cast<size_t>(frame.width) * frame.height);
    CUDA_CHECK(cudaMemcpy(host.data(), frame.d_data,
                          host.size() * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    return host;
}

}  // namespace

TEST(ISPPipelineTest, DefaultExecuteIsZeroCopyAndConsumesInput) {
    constexpr int width = 4;
    constexpr int height = 4;
    std::vector<uint16_t> black_levels = {64, 64, 64, 64};
    constexpr uint16_t black_level = 64; // for validation

    std::vector<uint16_t> original(width * height);
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = static_cast<uint16_t>(100 + i);
    }

    FrameBuffer input;
    input.width = width;
    input.height = height;
    input.channels = 1;
    input.format = PixelFormat::BAYER_RGGB;
    input.packing = PixelPacking::UNPACKED_U16;
    input.bit_depth = 10;
    input.allocate();
    CUDA_CHECK(cudaMemcpy(input.d_data, original.data(), input.sizeBytes(),
                          cudaMemcpyHostToDevice));

    ISPPipeline pipeline;
    pipeline.addBlock(createBlackLevelCorrection(black_levels, PixelFormat::BAYER_RGGB));

    const FrameBuffer result = pipeline.execute(input);
    const auto input_after = download(input);

    EXPECT_EQ(result.d_data, input.d_data);
    for (size_t i = 0; i < input_after.size(); ++i) {
        EXPECT_EQ(input_after[i], original[i] - black_level);
    }

    input.free();
}

TEST(ISPPipelineTest, PreservingExecuteKeepsInputAndIsRepeatable) {
    constexpr int width = 4;
    constexpr int height = 4;
    std::vector<uint16_t> black_levels = {64, 64, 64, 64};

    std::vector<uint16_t> original(width * height);
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = static_cast<uint16_t>(100 + i);
    }

    FrameBuffer input;
    input.width = width;
    input.height = height;
    input.channels = 1;
    input.format = PixelFormat::BAYER_RGGB;
    input.packing = PixelPacking::UNPACKED_U16;
    input.bit_depth = 10;
    input.allocate();
    CUDA_CHECK(cudaMemcpy(input.d_data, original.data(), input.sizeBytes(),
                          cudaMemcpyHostToDevice));

    ISPPipeline pipeline;
    pipeline.addBlock(createBlackLevelCorrection(black_levels, PixelFormat::BAYER_RGGB));

    const auto first = download(pipeline.executePreservingInput(input));
    const auto second = download(pipeline.executePreservingInput(input));

    EXPECT_EQ(download(input), original);
    EXPECT_EQ(second, first);
    input.free();
}

TEST(ISPPipelineTest, Performance_4K) {
    constexpr int width = 3840;
    constexpr int height = 2160;
    constexpr int num_iterations = 50; // Use 50 for full pipeline to save time
    
    FrameBuffer input;
    input.width = width;
    input.height = height;
    input.channels = 1;
    input.format = PixelFormat::BAYER_RGGB;
    input.packing = PixelPacking::UNPACKED_U16;
    input.bit_depth = 12;
    input.allocate();
    CUDA_CHECK(cudaMemset(input.d_data, 0, input.sizeBytes()));

    ISPPipeline pipeline;
    pipeline.addBlock(createRawUnpack());
    std::vector<uint16_t> black_levels = {64, 64, 64, 64};
    pipeline.addBlock(createBlackLevelCorrection(black_levels, PixelFormat::BAYER_RGGB));
    pipeline.addBlock(createDeadPixelCorrection(100, 100));
    
    std::vector<float> channel_lut(33 * 33, 1.0f);
    pipeline.addBlock(createLensShadingCorrection({channel_lut, channel_lut, channel_lut, channel_lut}, 33, 33, 12));
    
    pipeline.addBlock(createAutoWhiteBalance(12));
    pipeline.addBlock(createDemosaicOptimized(12));
    
    ColorCorrectionMatrix ccm;
    pipeline.addBlock(createColorCorrectionMatrix(ccm));
    pipeline.addBlock(createToneMapping(1.0f));
    pipeline.addBlock(createGammaCorrectionOptimized());
    
	YuvDenoiseConfig denoise_config;
	pipeline.addBlock(createRgbToYuv());
	pipeline.addBlock(createYuvDenoise(denoise_config));

    EdgeEnhancementConfig edge_config;
    pipeline.addBlock(createEdgeEnhancement(edge_config));
    
    pipeline.addBlock(createOutputPack());

    // Warm up
    FrameBuffer output = pipeline.executePreservingInput(input);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; ++i) {
        output = pipeline.executePreservingInput(input);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / num_iterations;

    std::cout << "\n============================================================\n";
    std::cout << " Full ISP Pipeline Performance Benchmark (4K: 3840x2160)\n";
    std::cout << "============================================================\n";
    std::cout << "Avg Time: " << avg_ms << " ms (" << 1000.0f / avg_ms << " FPS)\n";
    std::cout << "============================================================\n\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    input.free();
}
