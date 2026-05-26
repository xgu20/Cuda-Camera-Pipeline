#pragma once

#include "frame_buffer.h"
#include <cstdint>
#include <string>

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
    int          width        = 0;
    int          height       = 0;
    int          bit_depth    = 16;
    PixelFormat  bayer_format = PixelFormat::BAYER_RGGB;
    PixelPacking packing      = PixelPacking::UNPACKED_U16;
    uint16_t     black_level  = 0;
};

// Load a sensor config from a JSON file. Throws std::runtime_error on
// missing required fields, malformed JSON, or unknown enum strings.
SensorConfig loadSensorConfig(const std::string& path);

// Derive the JSON sidecar path from a raw path. Replaces the file extension
// with ".json", e.g. "data/foo.raw" -> "data/foo.json".
std::string defaultSidecarPath(const std::string& raw_path);
