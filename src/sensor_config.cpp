#include "sensor_config.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

enum class EnableMode { Auto, Enabled, Disabled };

EnableMode parseEnableMode(const json &root, const json *module,
					   const char *legacy_key) {
	if (module && module->contains("enabled")) {
		const auto &value = module->at("enabled");
		if (value.is_boolean())
			return value.get<bool>() ? EnableMode::Enabled : EnableMode::Disabled;
		if (value.is_string() && value.get<std::string>() == "auto")
			return EnableMode::Auto;
		throw std::runtime_error(std::string("sensor config: ") + legacy_key +
			" enabled must be true, false, or \"auto\"");
	}
	if (root.contains(legacy_key)) {
		const auto &value = root.at(legacy_key);
		if (value.is_boolean())
			return value.get<bool>() ? EnableMode::Enabled : EnableMode::Disabled;
		if (value.is_string() && value.get<std::string>() == "auto")
			return EnableMode::Auto;
		throw std::runtime_error(std::string("sensor config: ") + legacy_key +
			" must be true, false, or \"auto\"");
	}
	return EnableMode::Auto;
}

const json *findModule(const json &root, const char *name) {
	if (root.contains("modules") && root.at("modules").is_object() &&
		root.at("modules").contains(name))
		return &root.at("modules").at(name);
	if (root.contains(name)) return &root.at(name);
	return nullptr;
}

bool hasAnyParameter(const json *module,
					 std::initializer_list<const char *> names) {
	if (!module || !module->is_object()) return false;
	for (const char *name : names) {
		if (module->contains(name)) return true;
	}
	return false;
}

json loadOptionalGoldenTuning(const std::string &path) {
	if (path.empty()) return json::object();
	std::ifstream file(path);
	if (!file.is_open()) return json::object();
	json tuning;
	try {
		file >> tuning;
	} catch (const json::parse_error &e) {
		throw std::runtime_error("Malformed golden tuning JSON in " + path +
			": " + e.what());
	}
	return tuning;
}

YuvDenoiseConfig parseYuvDenoiseConfig(const json &j,
										YuvDenoiseConfig config = {}) {
	config.spatial_sigma = j.value("spatial_sigma", config.spatial_sigma);
	config.luma_range_sigma =
		j.value("luma_range_sigma", config.luma_range_sigma);
	config.chroma_range_sigma =
		j.value("chroma_range_sigma", config.chroma_range_sigma);
	config.luma_strength = j.value("luma_strength", config.luma_strength);
	config.chroma_strength = j.value("chroma_strength", config.chroma_strength);
	// Reuse the block's authoritative validation.
	(void)createYuvDenoise(config);
	return config;
}

EdgeEnhancementConfig parseEdgeEnhancementConfig(
	const json &j, EdgeEnhancementConfig config = {}) {
	config.strength = j.value("strength", config.strength);
	config.threshold = j.value("threshold", config.threshold);
	config.clamp_limit = j.value("clamp_limit", config.clamp_limit);
	(void)createEdgeEnhancement(config);
	return config;
}

bool resolveEnabled(EnableMode mode, bool has_tuning, const char *name) {
	if (mode == EnableMode::Disabled) return false;
	if (mode == EnableMode::Enabled && !has_tuning) {
		throw std::runtime_error(std::string("sensor config: ") + name +
			" is explicitly enabled but no sensor or golden tuning is available");
	}
	return has_tuning;
}

bool hasEnableSetting(const json &root, const json *module,
					  const char *legacy_key) {
	return (module && module->contains("enabled")) || root.contains(legacy_key);
}

bool resolveModuleEnabled(const json &sensor, const json &golden,
						  const char *name, const char *legacy_key,
						  bool sensor_has_tuning, bool golden_has_tuning) {
	const json *sensor_module = findModule(sensor, name);
	const json *golden_module = findModule(golden, name);
	EnableMode mode = EnableMode::Auto;
	if (hasEnableSetting(sensor, sensor_module, legacy_key)) {
		mode = parseEnableMode(sensor, sensor_module, legacy_key);
	} else if (golden_module && golden_module->contains("enabled")) {
		mode = parseEnableMode(golden, golden_module, legacy_key);
	}
	return resolveEnabled(mode, sensor_has_tuning || golden_has_tuning, name);
}

std::string tuningSource(bool sensor_has, bool golden_has) {
	return sensor_has ? "sensor" : (golden_has ? "golden" : "missing");
}

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

std::vector<uint16_t> parseBlackLevels(const json &j, uint16_t fallback) {
	if (j.contains("black_levels")) {
		const auto& arr = j.at("black_levels");
		if (!arr.is_array() || arr.size() != 4) {
			throw std::runtime_error("sensor config: black_levels must be an array of 4 numbers");
		}
		std::vector<uint16_t> levels(4);
		for (int i = 0; i < 4; ++i) {
			levels[i] = arr.at(i).get<uint16_t>();
		}
		return levels;
	}
	return {fallback, fallback, fallback, fallback};
}

std::vector<std::vector<uint16_t>> parseOecfLut(const json &j, int expected_size) {
	if (!j.is_array() || j.size() != 4) {
		throw std::runtime_error("sensor config: oecf_lut must be an array of 4 channel arrays");
	}
	std::vector<std::vector<uint16_t>> lut(4);
	for (size_t c = 0; c < 4; ++c) {
		const auto &ch_json = j.at(c);
		if (!ch_json.is_array() || ch_json.size() != static_cast<size_t>(expected_size)) {
			throw std::runtime_error("sensor config: oecf_lut channel " + std::to_string(c) +
									 " must be an array of size " + std::to_string(expected_size));
		}
		lut[c].resize(expected_size);
		for (size_t i = 0; i < static_cast<size_t>(expected_size); ++i) {
			lut[c][i] = ch_json.at(i).get<uint16_t>();
		}
	}
	return lut;
}
} // namespace

SensorConfig loadSensorConfig(const std::string &path,
							  const std::string &golden_tuning_path) {
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
	const json golden = loadOptionalGoldenTuning(golden_tuning_path);
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
	cfg.black_levels = parseBlackLevels(j, static_cast<uint16_t>(black_level));
	if (!j.contains("black_level") && !j.contains("black_levels")) {
		if (const json *golden_blc = findModule(golden, "black_level")) {
			cfg.black_levels = parseBlackLevels(*golden_blc, 0);
		}
	}
	const int default_white_level = (1 << cfg.bit_depth) - 1;
	const int white_level = j.value("white_level", default_white_level);
	const int default_threshold = std::min(8000, (2 << cfg.bit_depth) - 1);
	const int hot_pixel_threshold = j.value("hot_pixel_threshold", default_threshold);
	const int dead_pixel_threshold = j.value("dead_pixel_threshold", default_threshold);
	cfg.white_balance_gains = parseWhiteBalanceGains(
		j.value("white_balance_gains",
				json{{"r", 1.0f}, {"gr", 1.0f}, {"gb", 1.0f}, {"b", 1.0f}}));
	cfg.has_manual_white_balance_gains = j.contains("white_balance_gains");
	const json *sensor_wb = findModule(j, "white_balance");
	if (sensor_wb && sensor_wb->contains("mode")) {
		const std::string mode = sensor_wb->at("mode").get<std::string>();
		if (mode == "gray_world") {
			if (cfg.has_manual_white_balance_gains) {
				throw std::runtime_error(
					"sensor config: gray_world white balance cannot include "
					"white_balance_gains");
			}
		} else if (mode == "manual") {
			if (!cfg.has_manual_white_balance_gains) {
				throw std::runtime_error(
					"sensor config: manual white balance requires "
					"white_balance_gains");
			}
		} else {
			throw std::runtime_error(
				"sensor config: white_balance mode must be gray_world or manual");
		}
	}
	if (j.contains("color_correction_matrix")) {
		cfg.color_correction_matrix =
			parseColorCorrectionMatrix(j.at("color_correction_matrix"));
	} else if (const json *golden_ccm = findModule(golden, "color_correction")) {
		if (golden_ccm->contains("matrix")) {
			cfg.color_correction_matrix =
				parseColorCorrectionMatrix(golden_ccm->at("matrix"));
		}
	}

	if (j.contains("lsc_lut")) {
		cfg.lsc_grid_width = j.at("lsc_grid_width").get<int>();
		cfg.lsc_grid_height = j.at("lsc_grid_height").get<int>();
		if (cfg.lsc_grid_width < 2 || cfg.lsc_grid_height < 2) {
			throw std::runtime_error("sensor config: lsc_grid_width and lsc_grid_height must be at least 2");
		}
		cfg.lsc_lut = parseLscLut(j.at("lsc_lut"), cfg.lsc_grid_width * cfg.lsc_grid_height);
	}

	if (j.contains("oecf_lut")) {
		cfg.oecf_out_bit_depth = j.value("oecf_out_bit_depth", 16);
		int expected_size = 1 << cfg.bit_depth;
		cfg.oecf_lut = parseOecfLut(j.at("oecf_lut"), expected_size);
	}

	const auto resolve_standard = [&](const char *name, const char *legacy,
									 bool sensor_parameters = false) {
		const bool sensor_has = findModule(j, name) != nullptr || sensor_parameters;
		const bool golden_has = findModule(golden, name) != nullptr;
		cfg.tuning_sources[name] = tuningSource(sensor_has, golden_has);
		return resolveModuleEnabled(j, golden, name, legacy, sensor_has, golden_has);
	};
	cfg.enable_blc = resolve_standard(
		"black_level", "enable_blc", j.contains("black_level") || j.contains("black_levels"));
	cfg.enable_dpc = resolve_standard(
		"dead_pixel_correction", "enable_dpc",
		j.contains("hot_pixel_threshold") || j.contains("dead_pixel_threshold"));
	cfg.enable_wb = resolve_standard(
		"white_balance", "enable_wb",
		j.contains("white_balance_gains") ||
			(sensor_wb && sensor_wb->contains("mode")));
	cfg.enable_demosaic = resolve_standard("demosaic", "enable_demosaic");
	cfg.enable_ccm = resolve_standard(
		"color_correction", "enable_ccm", j.contains("color_correction_matrix"));
	cfg.enable_tone_mapping = resolve_standard(
		"tone_mapping", "enable_tone_mapping", j.contains("tone_mapping_exposure"));
	const json *golden_tone = findModule(golden, "tone_mapping");
	const json *sensor_tone = findModule(j, "tone_mapping");
	cfg.tone_mapping_exposure = golden_tone
		? golden_tone->value("exposure", 1.0f) : 1.0f;
	if (sensor_tone)
		cfg.tone_mapping_exposure =
			sensor_tone->value("exposure", cfg.tone_mapping_exposure);
	cfg.tone_mapping_exposure =
		j.value("tone_mapping_exposure", cfg.tone_mapping_exposure);
	cfg.enable_gamma = resolve_standard("gamma", "enable_gamma");
	const json *sensor_denoise = findModule(j, "yuv_denoise");
	const json *golden_denoise = findModule(golden, "yuv_denoise");
	const bool golden_has_denoise = hasAnyParameter(
		golden_denoise, {"spatial_sigma", "luma_range_sigma",
			"chroma_range_sigma", "luma_strength", "chroma_strength"});
	const bool sensor_has_denoise = hasAnyParameter(
		sensor_denoise, {"spatial_sigma", "luma_range_sigma",
			"chroma_range_sigma", "luma_strength", "chroma_strength"});
	if (golden_has_denoise) {
		cfg.yuv_denoise_config = parseYuvDenoiseConfig(*golden_denoise);
		cfg.yuv_denoise_tuning_source = "golden";
	}
	if (sensor_has_denoise) {
		cfg.yuv_denoise_config =
			parseYuvDenoiseConfig(*sensor_denoise, cfg.yuv_denoise_config);
		cfg.yuv_denoise_tuning_source = "sensor";
	}
	const bool has_denoise_tuning = sensor_has_denoise || golden_has_denoise;
	cfg.enable_yuv_denoise = resolveEnabled(
		parseEnableMode(j, sensor_denoise, "enable_yuv_denoise"),
		has_denoise_tuning, "yuv_denoise");

	const json *sensor_edge = findModule(j, "edge_enhancement");
	const json *golden_edge = findModule(golden, "edge_enhancement");
	const bool golden_has_edge = hasAnyParameter(
		golden_edge, {"strength", "threshold", "clamp_limit"});
	const bool sensor_has_edge = hasAnyParameter(
		sensor_edge, {"strength", "threshold", "clamp_limit"});
	if (golden_has_edge) {
		cfg.edge_enhancement_config = parseEdgeEnhancementConfig(*golden_edge);
		cfg.edge_enhancement_tuning_source = "golden";
	}
	if (sensor_has_edge) {
		cfg.edge_enhancement_config =
			parseEdgeEnhancementConfig(*sensor_edge, cfg.edge_enhancement_config);
		cfg.edge_enhancement_tuning_source = "sensor";
	}
	const bool has_edge_tuning = sensor_has_edge || golden_has_edge;
	cfg.enable_edge_enhancement = resolveEnabled(
		parseEnableMode(j, sensor_edge, "enable_edge_enhancement"),
		has_edge_tuning, "edge_enhancement");
	cfg.enable_output_pack = resolve_standard("output_pack", "enable_output_pack");
	cfg.enable_lsc = resolve_standard(
		"lens_shading", "enable_lsc", j.contains("lsc_lut"));
	const bool sensor_has_oecf = j.contains("oecf_lut");
	const json *golden_oecf = findModule(golden, "sensor_linearization");
	const bool golden_has_oecf =
		golden_oecf && golden_oecf->contains("lut");
	cfg.tuning_sources["sensor_linearization"] =
		tuningSource(sensor_has_oecf, golden_has_oecf);
	cfg.enable_oecf = resolveModuleEnabled(
		j, golden, "sensor_linearization", "enable_oecf",
		sensor_has_oecf, golden_has_oecf);

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
	for (auto bl : cfg.black_levels) {
		if (bl > max_code) {
			throw std::runtime_error(
				"sensor config: black_levels must be within the sensor code range");
		}
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
