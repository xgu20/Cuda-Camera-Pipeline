#include "sensor_config.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

PixelFormat parseBayerFormat(const std::string& s) {
    if (s == "RGGB") return PixelFormat::BAYER_RGGB;
    if (s == "BGGR") return PixelFormat::BAYER_BGGR;
    if (s == "GRBG") return PixelFormat::BAYER_GRBG;
    if (s == "GBRG") return PixelFormat::BAYER_GBRG;
    throw std::runtime_error("Unknown bayer_pattern: " + s +
                             " (expected RGGB | BGGR | GRBG | GBRG)");
}

PixelPacking parsePacking(const std::string& s) {
    if (s == "unpacked_u16") return PixelPacking::UNPACKED_U16;
    if (s == "mipi10")       return PixelPacking::PACKED_10_MIPI;
    throw std::runtime_error("Unknown packing: " + s +
                             " (expected unpacked_u16 | mipi10)");
}

}  // namespace

SensorConfig loadSensorConfig(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open sensor config: " + path);
    }

    json j;
    try {
        file >> j;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("Malformed JSON in " + path + ": " + e.what());
    }

    SensorConfig cfg;
    cfg.width        = j.at("width").get<int>();
    cfg.height       = j.at("height").get<int>();
    cfg.bit_depth    = j.value("bit_depth", 16);
    cfg.bayer_format = parseBayerFormat(j.at("bayer_pattern").get<std::string>());
    cfg.packing      = parsePacking(j.value("packing", std::string("unpacked_u16")));
    cfg.black_level  = static_cast<uint16_t>(j.value("black_level", 0));

    // Sanity checks
    if (cfg.width <= 0 || cfg.height <= 0) {
        throw std::runtime_error("sensor config: width/height must be positive");
    }
    if (cfg.packing == PixelPacking::PACKED_10_MIPI && cfg.width % 4 != 0) {
        throw std::runtime_error(
            "sensor config: mipi10 packing requires width to be a multiple of 4");
    }
    if (cfg.bit_depth < 1 || cfg.bit_depth > 16) {
        throw std::runtime_error("sensor config: bit_depth must be in [1, 16]");
    }

    return cfg;
}

std::string defaultSidecarPath(const std::string& raw_path) {
    auto dot = raw_path.find_last_of('.');
    auto sep = raw_path.find_last_of("/\\");
    if (dot == std::string::npos || (sep != std::string::npos && dot < sep)) {
        return raw_path + ".json";
    }
    return raw_path.substr(0, dot) + ".json";
}
