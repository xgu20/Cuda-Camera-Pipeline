// ============================================================================
// CUDA ISP Pipeline — Main Entry Point
//
// Usage:
//   ./cuda_isp <input.raw> [output.png]
//
// The sensor metadata
// (width/height/bit_depth/bayer_pattern/packing/black_level) is read from a
// JSON sidecar that lives next to <input.raw>. The default sidecar path
// replaces the raw's extension with ".json", e.g.
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

void printUsage(const char *prog) {
	printf("CUDA ISP Pipeline\n");
	printf("Usage: %s <input.raw> [output.png]\n\n", prog);
	printf("  input.raw   — Raw sensor file. A JSON sidecar with the same\n");
	printf(
		"                stem (e.g. input.json) describes its dimensions,\n");
	printf("                bit depth, bayer pattern, packing, etc.\n");
	printf("  output.png  — Output PNG file (default: output.png)\n");
}

int main(int argc, char *argv[]) {
	// Force eager module loading. The CUDA 11.7+ default ("lazy") delays
	// loading each kernel's fatbin until its first launch, which dumps tens
	// of ms onto whichever kernel happens to run first (BLC here, observed
	// ~80 ms for a kernel whose actual runtime is sub-ms). For a one-shot
	// CLI tool, paying the load cost up-front during context init gives a
	// much cleaner end-to-end timing. setenv(..., 0) preserves any
	// user-provided override.
	setenv("CUDA_MODULE_LOADING", "EAGER", 0);

	if (argc < 2) {
		printUsage(argv[0]);
		return 1;
	}

	const std::string input_path = argv[1];
	const std::string output_path = (argc >= 3) ? argv[2] : "output.png";
	const std::string sidecar_path = defaultSidecarPath(input_path);

	printf("=== CUDA ISP Pipeline ===\n");
	printf("  Input:   %s\n", input_path.c_str());
	printf("  Sidecar: %s\n", sidecar_path.c_str());
	printf("  Output:  %s\n\n", output_path.c_str());

	try {
		SensorConfig cfg = loadSensorConfig(sidecar_path);

		printf("  Resolution: %dx%d, bit_depth=%d, black_level=%u, "
			   "white_level=%u, hot_pixel_threshold=%u,"
			   " dead_pixel_threshold=%u\n\n",
			   cfg.width, cfg.height, cfg.bit_depth, cfg.black_level,
			   cfg.white_level, cfg.hot_pixel_threshold,
			   cfg.dead_pixel_threshold);

		// --- Query GPU info ---
		int device;
		CUDA_CHECK(cudaGetDevice(&device));
		cudaDeviceProp props;
		CUDA_CHECK(cudaGetDeviceProperties(&props, device));
		printf("  GPU: %s (SM %d.%d, %d SMs)\n\n", props.name, props.major,
			   props.minor, props.multiProcessorCount);

		// --- Load raw bytes to GPU ---
		FrameLoader loader;
		FrameBuffer input = loader.load(input_path, cfg);

		// --- Build the ISP pipeline ---
		ISPPipeline pipeline;
		pipeline.addBlock(
			createRawUnpack()); // packed -> uint16 (no-op if already unpacked)
		pipeline.addBlock(createBlackLevelCorrection(cfg.black_level));
		pipeline.addBlock(createDeadPixelCorrection(cfg.hot_pixel_threshold,
													cfg.dead_pixel_threshold));
		const uint16_t signal_max =
			static_cast<uint16_t>(cfg.white_level - cfg.black_level);
		if (cfg.has_manual_white_balance_gains) {
			pipeline.addBlock(createManualWhiteBalance(
				cfg.white_balance_gains, cfg.bit_depth, signal_max));
		} else {
			pipeline.addBlock(
				createAutoWhiteBalance(cfg.bit_depth, signal_max));
		}
		pipeline.addBlock(createDemosaicOptimized(
			cfg.bit_depth, cfg.black_level, cfg.white_level));
		pipeline.addBlock(
			createColorCorrectionMatrix(cfg.color_correction_matrix));
		pipeline.addBlock(createGammaCorrection());
		pipeline.addBlock(createOutputPack());

		// Optional steady-state benchmark: run the pipeline N times on the same
		// input (set BENCH_ITERS). The first frame pays the one-time buffer
		// allocation; subsequent frames reuse the pool and show steady-state
		// cost.
		const char *iters_env = getenv("BENCH_ITERS");
		const int iters =
			(iters_env && atoi(iters_env) > 0) ? atoi(iters_env) : 1;

		printf("Processing pipeline:\n");
		FrameBuffer result{};
		for (int it = 0; it < iters; ++it) {
			if (iters > 1)
				printf("--- frame %d/%d ---\n", it + 1, iters);
			result = pipeline.execute(input);
		}

		pipeline.printSummary();

		loader.savePNG(result, output_path);

		// `result` is a non-owning view into a pipeline-owned buffer; the
		// pipeline destructor frees the whole pool. We only own `input`.
		input.free();

		printf("Done!\n");
		return 0;
	} catch (const std::exception &e) {
		fprintf(stderr, "cuda_isp failed: %s\n", e.what());
		return 2;
	}
}
