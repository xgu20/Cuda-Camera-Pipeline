// ============================================================================
// CUDA ISP Pipeline — Main Entry Point
//
// Usage:
//   ./cuda_isp <input.raw> <width> <height> [output.png]
//
// Example:
//   ./cuda_isp data/test_rggb_1920x1080_10bit.raw 1920 1080 output.png
// ============================================================================

#include "blocks.h"
#include "frame_buffer.h"
#include "frame_loader.h"
#include "isp_pipeline.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

void printUsage(const char* prog) {
    printf("CUDA ISP Pipeline\n");
    printf("Usage: %s <input.raw> <width> <height> [output.png]\n\n", prog);
    printf("  input.raw   — Raw Bayer file (uint16_t, RGGB pattern)\n");
    printf("  width       — Image width in pixels\n");
    printf("  height      — Image height in pixels\n");
    printf("  output.png  — Output PNG file (default: output.png)\n");
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string input_path  = argv[1];
    const int         width       = std::atoi(argv[2]);
    const int         height      = std::atoi(argv[3]);
    const std::string output_path = (argc >= 5) ? argv[4] : "output.png";

    printf("=== CUDA ISP Pipeline ===\n");
    printf("  Input:  %s (%dx%d)\n", input_path.c_str(), width, height);
    printf("  Output: %s\n\n", output_path.c_str());

    // --- Query GPU info ---
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("  GPU: %s (SM %d.%d, %d SMs)\n\n",
           props.name, props.major, props.minor,
           props.multiProcessorCount);

    // --- Load raw Bayer data to GPU ---
    FrameLoader loader;
    FrameBuffer input = loader.loadRaw(input_path, width, height,
                                        PixelFormat::BAYER_RGGB);

    // --- Build the ISP pipeline ---
    ISPPipeline pipeline;
    pipeline.addBlock(createBlackLevelCorrection(64));   // Subtract BLC=64
    pipeline.addBlock(createDemosaic(10));                // 10-bit → float RGB
    pipeline.addBlock(createGammaCorrection());           // sRGB gamma
    pipeline.addBlock(createOutputPack());                // float → uint8

    printf("Processing pipeline:\n");
    FrameBuffer result = pipeline.execute(input);

    // --- Print summary ---
    pipeline.printSummary();

    // --- Save output ---
    loader.savePNG(result, output_path);

    // --- Cleanup ---
    input.free();
    // intermediates are freed by pipeline destructor

    printf("Done!\n");
    return 0;
}
