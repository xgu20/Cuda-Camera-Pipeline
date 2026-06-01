#include "isp_pipeline.h"
#include <cstdio>

ISPPipeline::ISPPipeline() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

ISPPipeline::~ISPPipeline() {
    // Free any intermediate buffers we allocated
    for (auto& buf : intermediates_) {
        buf.free();
    }
    cudaStreamDestroy(stream_);
}

void ISPPipeline::addBlock(std::unique_ptr<ISPBlock> block) {
    blocks_.push_back(std::move(block));
}

FrameBuffer ISPPipeline::execute(const FrameBuffer& input) {
    if (blocks_.empty()) {
        fprintf(stderr, "[ISPPipeline] Warning: no blocks in pipeline\n");
        return input;
    }

    // Reuse the buffer pool across calls. Only discard it when the input
    // geometry changes — otherwise every block reuses last frame's buffer
    // and pays no cudaMalloc cost (which dominates allocating blocks).
    if (input.width != pooled_w_ || input.height != pooled_h_) {
        for (auto& buf : intermediates_) {
            buf.free();
        }
        intermediates_.clear();
        pooled_w_ = input.width;
        pooled_h_ = input.height;
    }
    intermediates_.resize(blocks_.size());  // one slot per block (empty = unused)
    timings_.clear();

    FrameBuffer current = input;

    for (size_t i = 0; i < blocks_.size(); ++i) {
        auto& block = blocks_[i];

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Seed output with this slot's cached buffer. If it already holds
        // device memory from a previous frame, the block's allocate() is a
        // no-op (FrameBuffer::allocate returns early when d_data is set), so
        // we reuse the allocation instead of mallocing again.
        FrameBuffer output = intermediates_[i];

        CUDA_CHECK(cudaEventRecord(start, stream_));
        block->process(current, output, stream_);
        CUDA_CHECK(cudaEventRecord(stop, stream_));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        timings_.push_back(elapsed_ms);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        printf("  [%zu] %-30s  %.3f ms\n", i, block->name(), elapsed_ms);

        // Remember a freshly-allocated buffer for reuse next frame. In-place
        // blocks leave output aliasing current, so their slot stays empty.
        if (output.d_data != current.d_data) {
            intermediates_[i] = output;
        }

        current = output;
    }

    // The final buffer stays owned by the pool (for reuse next frame); the
    // caller gets a non-owning view and must not free it.
    return current;
}

void ISPPipeline::printSummary() const {
    printf("\n=== ISP Pipeline Summary ===\n");
    float total = 0.0f;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        float t = (i < timings_.size()) ? timings_[i] : 0.0f;
        printf("  [%zu] %-30s  %.3f ms\n", i, blocks_[i]->name(), t);
        total += t;
    }
    printf("  %-34s  %.3f ms\n", "TOTAL", total);
    printf("============================\n\n");
}
