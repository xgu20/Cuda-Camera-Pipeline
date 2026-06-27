#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "sensor_config.h"

namespace {

class SensorConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = std::filesystem::temp_directory_path() /
                ("cuda_isp_sensor_config_" +
                 std::to_string(reinterpret_cast<uintptr_t>(this)) + ".json");
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    void write(const std::string& json) {
        std::ofstream file(path_);
        file << json;
    }

    std::filesystem::path path_;
};

}  // namespace

TEST_F(SensorConfigTest, LoadsAndDefaultsWhiteLevel) {
    write(R"({
        "width": 1920,
        "height": 1080,
        "bit_depth": 10,
        "bayer_pattern": "RGGB",
        "packing": "unpacked_u16",
        "black_level": 64
    })");

    const SensorConfig cfg = loadSensorConfig(path_.string());
    EXPECT_EQ(cfg.black_level, 64);
    EXPECT_EQ(cfg.white_level, 1023);
}

TEST_F(SensorConfigTest, RejectsNegativeBlackLevel) {
    write(R"({
        "width": 1920,
        "height": 1080,
        "bit_depth": 10,
        "bayer_pattern": "RGGB",
        "black_level": -1
    })");

    EXPECT_THROW(loadSensorConfig(path_.string()), std::runtime_error);
}

TEST_F(SensorConfigTest, RejectsMipi10WithWrongBitDepth) {
    write(R"({
        "width": 1920,
        "height": 1080,
        "bit_depth": 12,
        "bayer_pattern": "RGGB",
        "packing": "mipi10"
    })");

    EXPECT_THROW(loadSensorConfig(path_.string()), std::runtime_error);
}

TEST_F(SensorConfigTest, RejectsWhiteLevelAtOrBelowBlackLevel) {
    write(R"({
        "width": 1920,
        "height": 1080,
        "bit_depth": 10,
        "bayer_pattern": "RGGB",
        "black_level": 64,
        "white_level": 64
    })");

    EXPECT_THROW(loadSensorConfig(path_.string()), std::runtime_error);
}

TEST_F(SensorConfigTest, LoadsColorCorrectionMatrix) {
    write(R"({
        "width": 1920,
        "height": 1080,
        "bit_depth": 10,
        "bayer_pattern": "RGGB",
        "color_correction_matrix": [
            1.1, -0.1, 0.0,
            0.0, 1.0, 0.0,
            0.0, -0.2, 1.2
        ]
    })");

    const SensorConfig cfg = loadSensorConfig(path_.string());
    EXPECT_FLOAT_EQ(cfg.color_correction_matrix.values[0], 1.1f);
    EXPECT_FLOAT_EQ(cfg.color_correction_matrix.values[1], -0.1f);
    EXPECT_FLOAT_EQ(cfg.color_correction_matrix.values[8], 1.2f);
}

TEST_F(SensorConfigTest, ParsesHotAndDeadPixelThresholds) {
    write(R"({
        "width": 1920,
        "height": 1080,
        "bit_depth": 10,
        "bayer_pattern": "RGGB",
        "hot_pixel_threshold": 500,
        "dead_pixel_threshold": 400
    })");

    const SensorConfig cfg = loadSensorConfig(path_.string());
    EXPECT_EQ(cfg.hot_pixel_threshold, 500);
    EXPECT_EQ(cfg.dead_pixel_threshold, 400);
}

TEST_F(SensorConfigTest, RejectsInvalidHotPixelThreshold) {
    write(R"({
        "width": 1920,
        "height": 1080,
        "bit_depth": 10,
        "bayer_pattern": "RGGB",
        "hot_pixel_threshold": 2048
    })");

    EXPECT_THROW(loadSensorConfig(path_.string()), std::runtime_error);
}

