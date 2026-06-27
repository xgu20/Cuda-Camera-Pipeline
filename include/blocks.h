#pragma once

#include "isp_block.h"
#include <cstdint>
#include <memory>
#include <vector>

// ============================================================================
// Factory functions for every ISPBlock implementation.
//
// Each .cu file in blocks/ defines a concrete ISPBlock subclass and exposes
// a `create*` function here so callers (main.cpp, tests, etc.) don't need
// to repeat `extern` declarations and can't get out of sync with the impl.
// ============================================================================
struct WhiteBalanceGains {
    float r  = 1.0f;
    float gr = 1.0f;
    float gb = 1.0f;
    float b  = 1.0f;
};

struct ColorCorrectionMatrix {
    float values[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
};

// --- Raw Unpack (packed sensor data -> unpacked uint16 Bayer) ---
// Default returns the best fully-implemented variant. The Naive / VecStore /
// VecRW factories below let benchmarks compare implementations side by side.
std::unique_ptr<ISPBlock> createRawUnpack();
std::unique_ptr<ISPBlock> createRawUnpackNaive();
std::unique_ptr<ISPBlock> createRawUnpackVecStore();
std::unique_ptr<ISPBlock> createRawUnpackVecRW();
std::unique_ptr<ISPBlock> createRawUnpackVecRWGrp4();

// --- Black Level Correction ---
std::unique_ptr<ISPBlock> createBlackLevelCorrection(uint16_t black_level);

// --- Dead Pixel Correction ---
std::unique_ptr<ISPBlock> createDeadPixelCorrection(uint16_t th_hot, uint16_t th_dead);

// --- Demosaic ---
std::unique_ptr<ISPBlock> createDemosaic(int bit_depth);
std::unique_ptr<ISPBlock> createDemosaic(int bit_depth, uint16_t black_level,
                                         uint16_t white_level);
std::unique_ptr<ISPBlock> createDemosaicOptimized(int bit_depth);
std::unique_ptr<ISPBlock> createDemosaicOptimized(int bit_depth, uint16_t black_level,
                                                  uint16_t white_level);

// --- Color Correction Matrix ---
std::unique_ptr<ISPBlock> createColorCorrectionMatrix(ColorCorrectionMatrix matrix);

// --- Gamma Correction ---
std::unique_ptr<ISPBlock> createGammaCorrection();
std::unique_ptr<ISPBlock> createGammaCorrectionOptimized();

// --- Output Packing ---
std::unique_ptr<ISPBlock> createOutputPack();

// --- Auto White Balance ---
std::unique_ptr<ISPBlock> createManualWhiteBalance(WhiteBalanceGains gains, int bit_depth,
                                                   uint16_t cut_off = 0);
std::unique_ptr<ISPBlock> createAutoWhiteBalance(int bit_depth,
                                                 uint16_t cut_off = 0);  // Gray World

// --- Lens Shading Correction ---
std::unique_ptr<ISPBlock> createLensShadingCorrection(
    const std::vector<std::vector<float>>& lut, int grid_width, int grid_height, int bit_depth);



