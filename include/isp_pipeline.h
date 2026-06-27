#pragma once

#include "isp_block.h"
#include <vector>
#include <memory>
#include <string>

// ============================================================================
// ISPPipeline — orchestrates a chain of ISPBlock stages.
//
// Usage:
//   ISPPipeline pipeline;
//   pipeline.addBlock(std::make_unique<BlackLevelCorrection>(64));
//   pipeline.addBlock(std::make_unique<Demosaic>());
//   pipeline.addBlock(std::make_unique<GammaCorrection>(2.2f));
//   pipeline.addBlock(std::make_unique<OutputPack>());
//   FrameBuffer result = pipeline.execute(input_frame);
// ============================================================================
class ISPPipeline {
public:
    ISPPipeline();
    ~ISPPipeline();

    // Add a processing block to the end of the pipeline
    void addBlock(std::unique_ptr<ISPBlock> block);

    // Execute all blocks in sequence with zero input copies.
    //
    // In-place blocks may modify input. Callers should treat input as consumed
    // after this call. The returned FrameBuffer may alias input when every
    // block runs in-place; otherwise it is a view into pipeline-owned memory.
    // The caller must NOT free the returned view.
    FrameBuffer execute(FrameBuffer& input);

    // Execute while preserving input. This performs one device-to-device copy
    // into a reusable staging buffer before running the blocks.
    //
    // The returned FrameBuffer is a non-owning view into pipeline-owned memory;
    // the caller must NOT free it.
    FrameBuffer executePreservingInput(const FrameBuffer& input);

    // Print pipeline summary (block names, timing)
    void printSummary() const;

private:
    std::vector<std::unique_ptr<ISPBlock>> blocks_;
    cudaStream_t stream_;

    // Persistent copy used only by executePreservingInput().
    FrameBuffer input_staging_;

    // Persistent per-block output buffer pool, reused across execute() calls.
    // Slot i holds the buffer block i allocated (empty for in-place blocks).
    // Buffers are only (re)allocated when the input geometry changes.
    std::vector<FrameBuffer> intermediates_;
    int pooled_w_ = 0;   // input geometry the current pool was sized for
    int pooled_h_ = 0;
    size_t pooled_input_bytes_ = 0;
    PixelFormat pooled_format_ = PixelFormat::BAYER_RGGB;
    PixelPacking pooled_packing_ = PixelPacking::UNPACKED_U16;
    int pooled_channels_ = 0;
    int pooled_bit_depth_ = 0;

    // Per-block timing (milliseconds)
    std::vector<float> timings_;

    void preparePool(const FrameBuffer& input);
    FrameBuffer executeFrom(FrameBuffer current);
};
