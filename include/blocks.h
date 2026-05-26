#pragma once

#include "isp_block.h"
#include <cstdint>
#include <memory>

// ============================================================================
// Factory functions for every ISPBlock implementation.
//
// Each .cu file in blocks/ defines a concrete ISPBlock subclass and exposes
// a `create*` function here so callers (main.cpp, tests, etc.) don't need
// to repeat `extern` declarations and can't get out of sync with the impl.
// ============================================================================

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
std::unique_ptr<ISPBlock> createBlackLevelCorrectionOptimized(uint16_t black_level);

// --- Demosaic ---
std::unique_ptr<ISPBlock> createDemosaic(int bit_depth);
std::unique_ptr<ISPBlock> createDemosaicOptimized(int bit_depth);

// --- Gamma Correction ---
std::unique_ptr<ISPBlock> createGammaCorrection();
std::unique_ptr<ISPBlock> createGammaCorrectionOptimized();

// --- Output Packing ---
std::unique_ptr<ISPBlock> createOutputPack();
