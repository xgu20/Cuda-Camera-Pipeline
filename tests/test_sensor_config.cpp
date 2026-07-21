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
                ("libreisp_sensor_config_" +
                 std::to_string(reinterpret_cast<uintptr_t>(this)) + ".json");
        golden_path_ = std::filesystem::temp_directory_path() /
                ("libreisp_golden_tuning_" +
                 std::to_string(reinterpret_cast<uintptr_t>(this)) + ".json");
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
        std::filesystem::remove(golden_path_, ec);
    }

    void write(const std::string& json) {
        std::ofstream file(path_);
        file << json;
    }

    void writeGolden(const std::string& json) {
        std::ofstream file(golden_path_);
        file << json;
    }

    std::filesystem::path path_;
    std::filesystem::path golden_path_;
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

TEST_F(SensorConfigTest, LoadsGrayWorldWhiteBalanceWithoutManualGains) {
    write(R"({
        "width": 1920,
        "height": 1080,
        "bit_depth": 10,
        "bayer_pattern": "RGGB",
        "modules": {
            "white_balance": {"enabled": true, "mode": "gray_world"}
        }
    })");

    const SensorConfig cfg = loadSensorConfig(path_.string(), "");
    EXPECT_TRUE(cfg.enable_wb);
    EXPECT_FALSE(cfg.has_manual_white_balance_gains);
}

TEST_F(SensorConfigTest, RejectsGrayWorldWithManualGains) {
    write(R"({
        "width": 1920,
        "height": 1080,
        "bit_depth": 10,
        "bayer_pattern": "RGGB",
        "modules": {"white_balance": {"mode": "gray_world"}},
        "white_balance_gains": {"r": 1.2, "gr": 1.0, "gb": 1.0, "b": 2.0}
    })");

    EXPECT_THROW(loadSensorConfig(path_.string(), ""), std::runtime_error);
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

TEST_F(SensorConfigTest, AutoUsesGoldenTuningWhenSensorTuningIsMissing) {
    write(R"({
        "width": 1920, "height": 1080, "bit_depth": 10,
        "bayer_pattern": "RGGB"
    })");
    writeGolden(R"({"modules": {
        "yuv_denoise": {
            "spatial_sigma": 1.7, "luma_range_sigma": 0.03,
            "chroma_range_sigma": 0.1, "luma_strength": 0.25,
            "chroma_strength": 0.8
        },
        "edge_enhancement": {
            "strength": 0.9, "threshold": 0.04, "clamp_limit": 0.08
        }
    }})");

    const SensorConfig cfg = loadSensorConfig(path_.string(), golden_path_.string());
    EXPECT_TRUE(cfg.enable_yuv_denoise);
    EXPECT_TRUE(cfg.enable_edge_enhancement);
    EXPECT_EQ(cfg.yuv_denoise_tuning_source, "golden");
    EXPECT_EQ(cfg.edge_enhancement_tuning_source, "golden");
    EXPECT_FLOAT_EQ(cfg.yuv_denoise_config.luma_strength, 0.25f);
    EXPECT_FLOAT_EQ(cfg.edge_enhancement_config.strength, 0.9f);
}

TEST_F(SensorConfigTest, SensorParametersOverrideGoldenAndExplicitFalseWins) {
    write(R"({
        "width": 1920, "height": 1080, "bit_depth": 10,
        "bayer_pattern": "RGGB",
        "yuv_denoise": {"luma_strength": 0.6},
        "edge_enhancement": {"enabled": false, "strength": 2.0}
    })");
    writeGolden(R"({"modules": {
        "yuv_denoise": {
            "spatial_sigma": 2.0, "luma_range_sigma": 0.025,
            "chroma_range_sigma": 0.12, "luma_strength": 0.4,
            "chroma_strength": 1.0
        },
        "edge_enhancement": {
            "strength": 1.5, "threshold": 0.02, "clamp_limit": 0.1
        }
    }})");

    const SensorConfig cfg = loadSensorConfig(path_.string(), golden_path_.string());
    EXPECT_TRUE(cfg.enable_yuv_denoise);
    EXPECT_FLOAT_EQ(cfg.yuv_denoise_config.luma_strength, 0.6f);
    EXPECT_FLOAT_EQ(cfg.yuv_denoise_config.spatial_sigma, 2.0f);
    EXPECT_EQ(cfg.yuv_denoise_tuning_source, "sensor");
    EXPECT_FALSE(cfg.enable_edge_enhancement);
}

TEST_F(SensorConfigTest, AutoDisablesWithoutAnyTuning) {
    write(R"({
        "width": 1920, "height": 1080, "bit_depth": 10,
        "bayer_pattern": "RGGB"
    })");
    const SensorConfig cfg = loadSensorConfig(path_.string(), "");
    EXPECT_FALSE(cfg.enable_yuv_denoise);
    EXPECT_FALSE(cfg.enable_edge_enhancement);
    EXPECT_EQ(cfg.yuv_denoise_tuning_source, "missing");
}

TEST_F(SensorConfigTest, ExplicitEnableWithoutTuningIsRejected) {
    write(R"({
        "width": 1920, "height": 1080, "bit_depth": 10,
        "bayer_pattern": "RGGB", "enable_yuv_denoise": true
    })");
    EXPECT_THROW(loadSensorConfig(path_.string(), ""), std::runtime_error);
}

TEST_F(SensorConfigTest, ProjectGoldenTuningResolvesEveryPipelineModule) {
    write(R"({
        "width": 1920, "height": 1080, "bit_depth": 10,
        "bayer_pattern": "RGGB"
    })");
    const auto project_golden =
        std::filesystem::path(__FILE__).parent_path().parent_path() /
        "config/golden_tuning.json";
    const SensorConfig cfg =
        loadSensorConfig(path_.string(), project_golden.string());

    EXPECT_TRUE(cfg.enable_blc);
    EXPECT_FALSE(cfg.enable_oecf); // no universal sensor OECF LUT
    EXPECT_TRUE(cfg.enable_dpc);
    EXPECT_TRUE(cfg.enable_lsc);
    EXPECT_TRUE(cfg.enable_wb);
    EXPECT_TRUE(cfg.enable_demosaic);
    EXPECT_TRUE(cfg.enable_ccm);
    EXPECT_TRUE(cfg.enable_tone_mapping);
    EXPECT_TRUE(cfg.enable_gamma);
    EXPECT_TRUE(cfg.enable_yuv_denoise);
    EXPECT_TRUE(cfg.enable_edge_enhancement);
    EXPECT_TRUE(cfg.enable_output_pack);
    EXPECT_EQ(cfg.tuning_sources.at("black_level"), "golden");
    EXPECT_EQ(cfg.tuning_sources.at("sensor_linearization"), "missing");
    EXPECT_EQ(cfg.tuning_sources.at("gamma"), "golden");
    EXPECT_EQ(cfg.yuv_denoise_tuning_source, "golden");
}
