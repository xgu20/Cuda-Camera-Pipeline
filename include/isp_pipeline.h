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

    // Execute all blocks in sequence, returning the final output.
    //
    // Ownership: the pipeline owns all intermediate buffers (including the
    // final output) and reuses them across calls. The returned FrameBuffer
    // is a non-owning view into a pipeline-owned buffer (or into `input` if
    // every block ran in-place); the caller must NOT free it. The buffers
    // stay valid until the next execute() with different geometry or until
    // the pipeline is destroyed.
    FrameBuffer execute(const FrameBuffer& input);

    // Print pipeline summary (block names, timing)
    void printSummary() const;

private:
    std::vector<std::unique_ptr<ISPBlock>> blocks_;
    cudaStream_t stream_;

    // Persistent per-block output buffer pool, reused across execute() calls.
    // Slot i holds the buffer block i allocated (empty for in-place blocks).
    // Buffers are only (re)allocated when the input geometry changes.
    std::vector<FrameBuffer> intermediates_;
    int pooled_w_ = 0;   // input geometry the current pool was sized for
    int pooled_h_ = 0;

    // Per-block timing (milliseconds)
    std::vector<float> timings_;
};
