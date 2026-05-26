// ============================================================================
// CUDA ISP Pipeline — Main Entry Point
//
// Usage:
//   ./cuda_isp <input.raw> [output.png]
//
// The sensor metadata (width/height/bit_depth/bayer_pattern/packing/black_level)
// is read from a JSON sidecar that lives next to <input.raw>. The default
// sidecar path replaces the raw's extension with ".json", e.g.
//   data/scene.raw  ->  data/scene.json
//
// Example:
//   ./cuda_isp data/test_rggb_1920x1080_10bit.raw output.png
// ============================================================================

#include "blocks.h"
#include "frame_buffer.h"
#include "frame_loader.h"
#include "isp_pipeline.h"
#include "sensor_config.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

void printUsage(const char* prog) {
    printf("CUDA ISP Pipeline\n");
    printf("Usage: %s <input.raw> [output.png]\n\n", prog);
    printf("  input.raw   — Raw sensor file. A JSON sidecar with the same\n");
    printf("                stem (e.g. input.json) describes its dimensions,\n");
    printf("                bit depth, bayer pattern, packing, etc.\n");
    printf("  output.png  — Output PNG file (default: output.png)\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string input_path  = argv[1];
    const std::string output_path = (argc >= 3) ? argv[2] : "output.png";
    const std::string sidecar_path = defaultSidecarPath(input_path);

    printf("=== CUDA ISP Pipeline ===\n");
    printf("  Input:   %s\n", input_path.c_str());
    printf("  Sidecar: %s\n", sidecar_path.c_str());
    printf("  Output:  %s\n\n", output_path.c_str());

    SensorConfig cfg;
    try {
        cfg = loadSensorConfig(sidecar_path);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to load sensor config: %s\n", e.what());
        return 2;
    }

    printf("  Resolution: %dx%d, bit_depth=%d, black_level=%u\n\n",
           cfg.width, cfg.height, cfg.bit_depth, cfg.black_level);

    // --- Query GPU info ---
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("  GPU: %s (SM %d.%d, %d SMs)\n\n",
           props.name, props.major, props.minor,
           props.multiProcessorCount);

    // --- Load raw bytes to GPU ---
    FrameLoader loader;
    FrameBuffer input = loader.load(input_path, cfg);

    // --- Build the ISP pipeline ---
    ISPPipeline pipeline;
    pipeline.addBlock(createRawUnpack());                       // packed -> uint16 (no-op if already unpacked)
    pipeline.addBlock(createBlackLevelCorrection(cfg.black_level));
    pipeline.addBlock(createDemosaic(cfg.bit_depth));
    pipeline.addBlock(createGammaCorrection());
    pipeline.addBlock(createOutputPack());

    printf("Processing pipeline:\n");
    FrameBuffer result = pipeline.execute(input);

    pipeline.printSummary();

    loader.savePNG(result, output_path);

    // The pipeline transfers ownership of the final buffer back to us when
    // it actually allocated; if every block ran in-place, result aliases
    // input and we must not double-free.
    if (result.d_data != input.d_data) {
        result.free();
    }
    input.free();
    // pipeline destructor frees any remaining intermediates

    printf("Done!\n");
    return 0;
}
