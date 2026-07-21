// ============================================================================
// LibreCudaISP — Main Entry Point
//
// Usage:
//   ./libreisp <input.raw> [output.png] [--config sensor.json]
//
// The sensor metadata
// (width/height/bit_depth/bayer_pattern/packing/black_level) is read from a
// JSON sidecar that lives next to <input.raw>. The default sidecar path
// replaces the raw's extension with ".json", e.g.
//   data/scene.raw  ->  data/scene.json
//
// Example:
//   ./libreisp data/test_rggb_1920x1080_10bit.raw output.png
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
#include <vector>

void printUsage(const char *prog) {
	printf("LibreCudaISP\n");
	printf("Usage: %s <input.raw> [output.png] [--config sensor.json]\n\n", prog);
	printf("  input.raw   — Raw sensor file. A JSON sidecar with the same\n");
	printf(
		"                stem (e.g. input.json) describes its dimensions,\n");
	printf("                bit depth, bayer pattern, packing, etc., unless\n");
	printf("                --config is provided.\n");
	printf("  output.png  — Output PNG file (default: output.png)\n");
	printf("  -c, --config — Explicit JSON config, reusable across RAW files\n");
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

	std::string config_path;
	std::vector<std::string> positional;
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		if (arg == "-h" || arg == "--help") {
			printUsage(argv[0]);
			return 0;
		}
		if (arg == "-c" || arg == "--config") {
			if (++i >= argc) {
				fprintf(stderr, "%s requires a JSON path\n", arg.c_str());
				return 1;
			}
			config_path = argv[i];
			continue;
		}
		if (arg.rfind("--config=", 0) == 0) {
			config_path = arg.substr(std::string("--config=").size());
			if (config_path.empty()) {
				fprintf(stderr, "--config requires a JSON path\n");
				return 1;
			}
			continue;
		}
		if (!arg.empty() && arg[0] == '-') {
			fprintf(stderr, "Unknown option: %s\n", arg.c_str());
			return 1;
		}
		positional.push_back(arg);
	}

	if (positional.empty() || positional.size() > 2) {
		printUsage(argv[0]);
		return 1;
	}

	const std::string input_path = positional[0];
	const std::string output_path =
		(positional.size() == 2) ? positional[1] : "output.png";
	const std::string sidecar_path =
		config_path.empty() ? defaultSidecarPath(input_path) : config_path;
	const char *golden_env = getenv("GOLDEN_TUNING_FILE");
	const std::string golden_tuning_path =
		golden_env ? golden_env : "config/golden_tuning.json";

	printf("=== LibreCudaISP ===\n");
	printf("  Input:   %s\n", input_path.c_str());
	printf("  Sensor config: %s%s\n", sidecar_path.c_str(),
		   config_path.empty() ? " (sidecar)" : " (explicit)");
	printf("  Golden tuning: %s\n", golden_tuning_path.c_str());
	printf("  Output:  %s\n\n", output_path.c_str());

	try {
		SensorConfig cfg = loadSensorConfig(sidecar_path, golden_tuning_path);

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

		auto raw_unpack = createRawUnpack();
		pipeline.addBlock(std::move(raw_unpack));

		auto blc = createBlackLevelCorrection(cfg.black_levels, cfg.bayer_format);
		blc->setBypass(!cfg.enable_blc);
		pipeline.addBlock(std::move(blc));

		auto oecf = createSensorLinearization(cfg.oecf_lut, cfg.bayer_format, cfg.oecf_out_bit_depth);
		oecf->setBypass(!cfg.enable_oecf);
		pipeline.addBlock(std::move(oecf));

		auto dpc = createDeadPixelCorrection(cfg.hot_pixel_threshold,
											 cfg.dead_pixel_threshold);
		dpc->setBypass(!cfg.enable_dpc);
		pipeline.addBlock(std::move(dpc));

		auto lsc = createLensShadingCorrection(cfg.lsc_lut, cfg.lsc_grid_width, cfg.lsc_grid_height, cfg.bit_depth);
		lsc->setBypass(!cfg.enable_lsc);
		pipeline.addBlock(std::move(lsc));

		const uint16_t signal_max =
			static_cast<uint16_t>(cfg.white_level - cfg.black_level);
		std::unique_ptr<ISPBlock> wb;
		if (cfg.has_manual_white_balance_gains) {
			wb = createManualWhiteBalance(
				cfg.white_balance_gains, cfg.bit_depth, signal_max);
		} else {
			wb = createAutoWhiteBalance(cfg.bit_depth, signal_max);
		}
		wb->setBypass(!cfg.enable_wb);
		pipeline.addBlock(std::move(wb));

		auto demosaic = createDemosaicOptimized(
			cfg.bit_depth, cfg.black_level, cfg.white_level);
		demosaic->setBypass(!cfg.enable_demosaic);
		pipeline.addBlock(std::move(demosaic));

		auto ccm = createColorCorrectionMatrix(cfg.color_correction_matrix);
		ccm->setBypass(!cfg.enable_ccm);
		pipeline.addBlock(std::move(ccm));

		auto tone_mapping = createToneMapping(cfg.tone_mapping_exposure);
		tone_mapping->setBypass(!cfg.enable_tone_mapping);
		pipeline.addBlock(std::move(tone_mapping));

		auto gamma = createGammaCorrection();
		gamma->setBypass(!cfg.enable_gamma);
		pipeline.addBlock(std::move(gamma));

		printf("  YuvDenoise: %s (tuning=%s)\n",
			   cfg.enable_yuv_denoise ? "enabled" : "disabled",
			   cfg.yuv_denoise_tuning_source.c_str());
		printf("  EdgeEnhancement: %s (tuning=%s)\n\n",
			   cfg.enable_edge_enhancement ? "enabled" : "disabled",
			   cfg.edge_enhancement_tuning_source.c_str());

		// Color conversion is independent from the optional algorithms. Both
		// processing blocks now preserve YUV format, so either can be bypassed.
		auto rgb_to_yuv = createRgbToYuv();
		rgb_to_yuv->setBypass(!cfg.enable_yuv_denoise &&
							 !cfg.enable_edge_enhancement);
		pipeline.addBlock(std::move(rgb_to_yuv));

		auto yuv_denoise = createYuvDenoise(cfg.yuv_denoise_config);
		yuv_denoise->setBypass(!cfg.enable_yuv_denoise);
		pipeline.addBlock(std::move(yuv_denoise));

		auto edge_enhancement =
			createEdgeEnhancement(cfg.edge_enhancement_config);
		edge_enhancement->setBypass(!cfg.enable_edge_enhancement);
		pipeline.addBlock(std::move(edge_enhancement));

		auto output_pack = createOutputPack();
		output_pack->setBypass(!cfg.enable_output_pack);
		pipeline.addBlock(std::move(output_pack));

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
		fprintf(stderr, "libreisp failed: %s\n", e.what());
		return 2;
	}
}
