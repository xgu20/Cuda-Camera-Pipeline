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

    // Execute all blocks in sequence, returning the final output
    FrameBuffer execute(const FrameBuffer& input);

    // Print pipeline summary (block names, timing)
    void printSummary() const;

private:
    std::vector<std::unique_ptr<ISPBlock>> blocks_;
    cudaStream_t stream_;

    // Intermediate buffers (managed internally)
    std::vector<FrameBuffer> intermediates_;

    // Per-block timing (milliseconds)
    std::vector<float> timings_;
};
