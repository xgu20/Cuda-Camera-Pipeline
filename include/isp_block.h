#pragma once

#include "frame_buffer.h"
#include <cuda_runtime.h>

// ============================================================================
// ISPBlock — abstract interface for every processing stage in the pipeline.
//
// To add a new block:
//   1. Create a new .cu file in blocks/
//   2. Inherit from ISPBlock
//   3. Implement name() and process()
//   4. Register it in main.cpp with pipeline.addBlock(...)
// ============================================================================
class ISPBlock {
public:
    virtual ~ISPBlock() = default;

    // Human-readable name (used for logging / profiling)
    virtual const char* name() const = 0;

    // Process a frame.
    //   - input:  the source FrameBuffer (device memory)
    //   - output: the destination FrameBuffer (block must allocate if needed)
    //   - stream: CUDA stream for async execution
    //
    // For in-place operations, output can alias input (output = input).
    // For format-changing operations (e.g. demosaic), the block must allocate
    // a new output buffer with the correct dimensions/format.
    virtual void process(const FrameBuffer& input, FrameBuffer& output,
                         cudaStream_t stream) = 0;

    // Bypass getters / setters
    bool isBypass() const { return bypass_; }
    void setBypass(bool bypass) { bypass_ = bypass; }

private:
    bool bypass_ = false;
};
