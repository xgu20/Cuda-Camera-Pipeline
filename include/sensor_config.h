#pragma once

#include "blocks.h"
#include "frame_buffer.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// ============================================================================
// SensorConfig — metadata describing how to interpret a raw sensor file.
//
// Loaded from a JSON sidecar that lives next to the .raw file. Example:
//
//   {
//     "width": 1920,
//     "height": 1080,
//     "bit_depth": 10,
//     "bayer_pattern": "RGGB",      // RGGB | BGGR | GRBG | GBRG
//     "packing": "mipi10",          // unpacked_u16 | mipi10
//     "black_level": 64
//   }
//
// Optional fields fall back to sensible defaults (bit_depth=16,
// packing="unpacked_u16", black_level=0). `width`, `height`, and
// `bayer_pattern` are required.
// ============================================================================
struct SensorConfig {
	int width = 0;
	int height = 0;
	int bit_depth = 16;
	PixelFormat bayer_format = PixelFormat::BAYER_RGGB;
	PixelPacking packing = PixelPacking::UNPACKED_U16;
	uint16_t black_level = 0;
	std::vector<uint16_t> black_levels = {0, 0, 0, 0}; // R, Gr, Gb, B
	uint16_t white_level = 65535;
	uint16_t hot_pixel_threshold = 8000;
	uint16_t dead_pixel_threshold = 8000;
	WhiteBalanceGains white_balance_gains = {1.0f, 1.0f, 1.0f, 1.0f};
	bool has_manual_white_balance_gains = false;
	ColorCorrectionMatrix color_correction_matrix;
	YuvDenoiseConfig yuv_denoise_config;
	EdgeEnhancementConfig edge_enhancement_config;
	std::string yuv_denoise_tuning_source = "missing";
	std::string edge_enhancement_tuning_source = "missing";
	std::map<std::string, std::string> tuning_sources;

	// LSC grid and lookup tables (default is identity/flat)
	int lsc_grid_width = 2;
	int lsc_grid_height = 2;
	std::vector<std::vector<float>> lsc_lut = {
		{1.0f, 1.0f, 1.0f, 1.0f}, // R
		{1.0f, 1.0f, 1.0f, 1.0f}, // Gr
		{1.0f, 1.0f, 1.0f, 1.0f}, // Gb
		{1.0f, 1.0f, 1.0f, 1.0f}  // B
	};

	// OECF (Sensor Linearization) lookup table
	int oecf_out_bit_depth = 16;
	std::vector<std::vector<uint16_t>> oecf_lut; // Empty if not provided, otherwise 4 channels

	// Pipeline stage toggles (default to true)
	bool enable_blc = true;
	bool enable_dpc = true;
	bool enable_wb = true;
	bool enable_demosaic = true;
	bool enable_ccm = true;
	bool enable_tone_mapping = true;
	float tone_mapping_exposure = 1.0f;
	bool enable_gamma = true;
	bool enable_yuv_denoise = false;
	bool enable_edge_enhancement = false;
	bool enable_output_pack = true;
	bool enable_lsc = true;
	bool enable_oecf = true;
};

// Load a sensor config from a JSON file. Throws std::runtime_error on
// missing required fields, malformed JSON, or unknown enum strings.
SensorConfig loadSensorConfig(
	const std::string &path,
	const std::string &golden_tuning_path = "config/golden_tuning.json");

// Derive the JSON sidecar path from a raw path. Replaces the file extension
// with ".json", e.g. "data/foo.raw" -> "data/foo.json".
std::string defaultSidecarPath(const std::string &raw_path);
