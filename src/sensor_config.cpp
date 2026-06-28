#include "sensor_config.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

PixelFormat parseBayerFormat(const std::string &s) {
	if (s == "RGGB")
		return PixelFormat::BAYER_RGGB;
	if (s == "BGGR")
		return PixelFormat::BAYER_BGGR;
	if (s == "GRBG")
		return PixelFormat::BAYER_GRBG;
	if (s == "GBRG")
		return PixelFormat::BAYER_GBRG;
	throw std::runtime_error("Unknown bayer_pattern: " + s +
							 " (expected RGGB | BGGR | GRBG | GBRG)");
}

PixelPacking parsePacking(const std::string &s) {
	if (s == "unpacked_u8")
		return PixelPacking::UNPACKED_U8;
	if (s == "unpacked_u16")
		return PixelPacking::UNPACKED_U16;
	if (s == "mipi10")
		return PixelPacking::PACKED_10_MIPI;
	throw std::runtime_error("Unknown packing: " + s +
							 " (expected unpacked_u8 | unpacked_u16 | mipi10)");
}

WhiteBalanceGains parseWhiteBalanceGains(const json &j) {
	float r = j.value("r", 1.0f);
	float gr = j.value("gr", 1.0f);
	float gb = j.value("gb", 1.0f);
	float b = j.value("b", 1.0f);
	return WhiteBalanceGains{r, gr, gb, b};
}

ColorCorrectionMatrix parseColorCorrectionMatrix(const json &j) {
	if (!j.is_array() || j.size() != 9) {
		throw std::runtime_error("sensor config: color_correction_matrix must "
								 "be a flat array of 9 numbers");
	}

	ColorCorrectionMatrix matrix;
	for (size_t i = 0; i < 9; ++i) {
		matrix.values[i] = j.at(i).get<float>();
		if (!std::isfinite(matrix.values[i])) {
			throw std::runtime_error(
				"sensor config: color_correction_matrix values must be finite");
		}
	}
	return matrix;
}

std::vector<std::vector<float>> parseLscLut(const json &j, int expected_size) {
	if (!j.is_array() || j.size() != 4) {
		throw std::runtime_error("sensor config: lsc_lut must be an array of 4 channel arrays");
	}
	std::vector<std::vector<float>> lut(4);
	for (size_t c = 0; c < 4; ++c) {
		const auto &ch_json = j.at(c);
		if (!ch_json.is_array() || ch_json.size() != static_cast<size_t>(expected_size)) {
			throw std::runtime_error("sensor config: lsc_lut channel " + std::to_string(c) +
									 " must be an array of size " + std::to_string(expected_size));
		}
		lut[c].resize(expected_size);
		for (size_t i = 0; i < static_cast<size_t>(expected_size); ++i) {
			lut[c][i] = ch_json.at(i).get<float>();
			if (!std::isfinite(lut[c][i]) || lut[c][i] <= 0.0f) {
				throw std::runtime_error("sensor config: lsc_lut values must be finite and positive");
			}
		}
	}
	return lut;
}
} // namespace

SensorConfig loadSensorConfig(const std::string &path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open sensor config: " + path);
	}

	json j;
	try {
		file >> j;
	} catch (const json::parse_error &e) {
		throw std::runtime_error("Malformed JSON in " + path + ": " + e.what());
	}

	SensorConfig cfg;
	cfg.width = j.at("width").get<int>();
	cfg.height = j.at("height").get<int>();
	cfg.bit_depth = j.value("bit_depth", 16);
	cfg.bayer_format =
		parseBayerFormat(j.at("bayer_pattern").get<std::string>());
	cfg.packing = parsePacking(j.value("packing", std::string("unpacked_u16")));
	if (cfg.bit_depth < 1 || cfg.bit_depth > 16) {
		throw std::runtime_error("sensor config: bit_depth must be in [1, 16]");
	}
	const int black_level = j.value("black_level", 0);
	const int default_white_level = (1 << cfg.bit_depth) - 1;
	const int white_level = j.value("white_level", default_white_level);
	const int default_threshold = std::min(8000, (2 << cfg.bit_depth) - 1);
	const int hot_pixel_threshold = j.value("hot_pixel_threshold", default_threshold);
	const int dead_pixel_threshold = j.value("dead_pixel_threshold", default_threshold);
	cfg.white_balance_gains = parseWhiteBalanceGains(
		j.value("white_balance_gains",
				json{{"r", 1.0f}, {"gr", 1.0f}, {"gb", 1.0f}, {"b", 1.0f}}));
	cfg.has_manual_white_balance_gains = j.contains("white_balance_gains");
	if (j.contains("color_correction_matrix")) {
		cfg.color_correction_matrix =
			parseColorCorrectionMatrix(j.at("color_correction_matrix"));
	}

	if (j.contains("lsc_lut")) {
		cfg.lsc_grid_width = j.at("lsc_grid_width").get<int>();
		cfg.lsc_grid_height = j.at("lsc_grid_height").get<int>();
		if (cfg.lsc_grid_width < 2 || cfg.lsc_grid_height < 2) {
			throw std::runtime_error("sensor config: lsc_grid_width and lsc_grid_height must be at least 2");
		}
		cfg.lsc_lut = parseLscLut(j.at("lsc_lut"), cfg.lsc_grid_width * cfg.lsc_grid_height);
	}

	cfg.enable_blc = j.value("enable_blc", true);
	cfg.enable_dpc = j.value("enable_dpc", true);
	cfg.enable_wb = j.value("enable_wb", true);
	cfg.enable_demosaic = j.value("enable_demosaic", true);
	cfg.enable_ccm = j.value("enable_ccm", true);
	cfg.enable_gamma = j.value("enable_gamma", true);
	cfg.enable_output_pack = j.value("enable_output_pack", true);
	cfg.enable_lsc = j.value("enable_lsc", true);

	// Sanity checks
	if (cfg.width <= 0 || cfg.height <= 0) {
		throw std::runtime_error(
			"sensor config: width/height must be positive");
	}
	if ((cfg.width & 1) != 0 || (cfg.height & 1) != 0) {
		throw std::runtime_error(
			"sensor config: Bayer width/height must be even");
	}
	if (cfg.packing == PixelPacking::PACKED_10_MIPI && cfg.width % 4 != 0) {
		throw std::runtime_error("sensor config: mipi10 packing requires width "
								 "to be a multiple of 4");
	}
	if (cfg.packing == PixelPacking::PACKED_10_MIPI && cfg.bit_depth != 10) {
		throw std::runtime_error(
			"sensor config: mipi10 packing requires bit_depth == 10");
	}
	if (cfg.packing == PixelPacking::UNPACKED_U8 && cfg.bit_depth > 8) {
		throw std::runtime_error(
			"sensor config: unpacked_u8 packing requires bit_depth <= 8 (got " +
			std::to_string(cfg.bit_depth) + ")");
	}
	const int max_code = (1 << cfg.bit_depth) - 1;
	if (black_level < 0 || black_level > max_code) {
		throw std::runtime_error(
			"sensor config: black_level must be within the sensor code range");
	}
	if (white_level <= black_level || white_level > max_code) {
		throw std::runtime_error(
			"sensor config: white_level must be greater than black_level and "
			"within the sensor code range");
	}
	if (hot_pixel_threshold < 0 || hot_pixel_threshold >= (2 << cfg.bit_depth) ||
		dead_pixel_threshold < 0 || dead_pixel_threshold >= (2 << cfg.bit_depth)) {
		throw std::runtime_error(
			"sensor config: hot_pixel_threshold and dead_pixel_threshold "
			"must be smaller than 2 << bitdepth");
	}
	const auto valid_gain = [](float gain) {
		return std::isfinite(gain) && gain > 0.0f;
	};
	if (!valid_gain(cfg.white_balance_gains.r) ||
		!valid_gain(cfg.white_balance_gains.gr) ||
		!valid_gain(cfg.white_balance_gains.gb) ||
		!valid_gain(cfg.white_balance_gains.b)) {
		throw std::runtime_error(
			"sensor config: white_balance_gains must be finite and positive");
	}

	cfg.black_level = static_cast<uint16_t>(black_level);
	cfg.white_level = static_cast<uint16_t>(white_level);
	cfg.hot_pixel_threshold = static_cast<uint16_t>(hot_pixel_threshold);
	cfg.dead_pixel_threshold = static_cast<uint16_t>(dead_pixel_threshold);
	return cfg;
}

std::string defaultSidecarPath(const std::string &raw_path) {
	auto dot = raw_path.find_last_of('.');
	auto sep = raw_path.find_last_of("/\\");
	if (dot == std::string::npos || (sep != std::string::npos && dot < sep)) {
		return raw_path + ".json";
	}
	return raw_path.substr(0, dot) + ".json";
}
