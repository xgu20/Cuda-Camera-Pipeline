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

    // Free previous intermediates
    for (auto& buf : intermediates_) {
        buf.free();
    }
    intermediates_.clear();
    timings_.clear();

    FrameBuffer current = input;

    for (size_t i = 0; i < blocks_.size(); ++i) {
        auto& block = blocks_[i];

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        FrameBuffer output{};

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

        // If the block allocated a new buffer (output != input), track it
        if (output.d_data != current.d_data) {
            intermediates_.push_back(output);
        }

        current = output;
    }

    // Transfer ownership of the final buffer to the caller, so it survives
    // the pipeline's lifetime. The caller is responsible for calling .free()
    // on the returned buffer iff its d_data differs from the input's
    // d_data (i.e. the pipeline actually allocated something). If every
    // block ran in-place, `current` is just a view of `input` and must not
    // be freed by the caller.
    if (!intermediates_.empty() &&
        intermediates_.back().d_data == current.d_data) {
        intermediates_.pop_back();
    }
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
