#include "isp_pipeline.h"
#include <cstdio>

ISPPipeline::ISPPipeline() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

ISPPipeline::~ISPPipeline() {
    input_staging_.free();
    // Free any intermediate buffers we allocated
    for (auto& buf : intermediates_) {
        buf.free();
    }
    cudaStreamDestroy(stream_);
}

void ISPPipeline::addBlock(std::unique_ptr<ISPBlock> block) {
    blocks_.push_back(std::move(block));
}

void ISPPipeline::preparePool(const FrameBuffer& input) {
    const size_t input_bytes = input.sizeBytes();
    const bool layout_changed =
        input.width != pooled_w_ ||
        input.height != pooled_h_ ||
        input_bytes != pooled_input_bytes_ ||
        input.format != pooled_format_ ||
        input.packing != pooled_packing_ ||
        input.channels != pooled_channels_ ||
        input.bit_depth != pooled_bit_depth_;

    // Reuse the buffer pool across calls. Discard it when any part of the input
    // layout changes, since equal geometry can still have a different packing
    // or allocation size.
    if (layout_changed) {
        input_staging_.free();
        for (auto& buf : intermediates_) {
            buf.free();
        }
        intermediates_.clear();
        pooled_w_ = input.width;
        pooled_h_ = input.height;
        pooled_input_bytes_ = input_bytes;
        pooled_format_ = input.format;
        pooled_packing_ = input.packing;
        pooled_channels_ = input.channels;
        pooled_bit_depth_ = input.bit_depth;
    }
    intermediates_.resize(blocks_.size());  // one slot per block (empty = unused)
    timings_.clear();
}

FrameBuffer ISPPipeline::execute(FrameBuffer& input) {
    preparePool(input);
    return executeFrom(input);
}

FrameBuffer ISPPipeline::executePreservingInput(const FrameBuffer& input) {
    preparePool(input);

    input_staging_.width = input.width;
    input_staging_.height = input.height;
    input_staging_.channels = input.channels;
    input_staging_.format = input.format;
    input_staging_.packing = input.packing;
    input_staging_.bit_depth = input.bit_depth;
    input_staging_.allocate();
    CUDA_CHECK(cudaMemcpyAsync(input_staging_.d_data, input.d_data, input.sizeBytes(),
                               cudaMemcpyDeviceToDevice, stream_));

    return executeFrom(input_staging_);
}

FrameBuffer ISPPipeline::executeFrom(FrameBuffer current) {
    if (blocks_.empty()) {
        fprintf(stderr, "[ISPPipeline] Warning: no blocks in pipeline\n");
        return current;
    }

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

        float elapsed_ms = 0.0f;
        if (block->isBypass()) {
            output = current;
            timings_.push_back(0.0f);
            printf("  [%zu] %-30s  (Bypassed)\n", i, block->name());
        } else {
            CUDA_CHECK(cudaEventRecord(start, stream_));
            block->process(current, output, stream_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop, stream_));
            CUDA_CHECK(cudaEventSynchronize(stop));

            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            timings_.push_back(elapsed_ms);

            printf("  [%zu] %-30s  %.3f ms\n", i, block->name(), elapsed_ms);
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // Remember a freshly-allocated buffer for reuse next frame. In-place
        // blocks leave output aliasing current, so their slot stays empty.
        if (output.d_data != current.d_data) {
            intermediates_[i] = output;
        }

        current = output;
    }

    // The final buffer stays owned by the pool (for reuse next frame); the
    // caller gets a non-owning view and must not free it.
    if (timing_sums_.size() != timings_.size()) {
        timing_sums_.assign(timings_.size(), 0.0);
        timing_samples_ = 0;
    }
    for (size_t i = 0; i < timings_.size(); ++i) {
        timing_sums_[i] += timings_[i];
    }
    ++timing_samples_;
    return current;
}

void ISPPipeline::resetTimingStatistics() {
    timing_sums_.assign(blocks_.size(), 0.0);
    timing_samples_ = 0;
}

void ISPPipeline::printAverageSummary() const {
    if (timing_samples_ == 0) {
        printf("\n=== ISP Average Pipeline Summary ===\n  No timing samples\n"
               "====================================\n\n");
        return;
    }

    printf("\n=== ISP Average Pipeline Summary (%zu frames) ===\n",
           timing_samples_);
    double total = 0.0;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        if (blocks_[i]->isBypass()) {
            printf("  [%zu] %-30s  (Bypassed)\n", i, blocks_[i]->name());
        } else {
            const double average = timing_sums_[i] / timing_samples_;
            printf("  [%zu] %-30s  %.3f ms\n", i, blocks_[i]->name(), average);
            total += average;
        }
    }
    printf("  %-34s  %.3f ms (%.1f FPS)\n", "AVERAGE TOTAL", total,
           total > 0.0 ? 1000.0 / total : 0.0);
    printf("===============================================\n\n");
}

void ISPPipeline::printSummary() const {
    printf("\n=== ISP Pipeline Summary ===\n");
    float total = 0.0f;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        float t = (i < timings_.size()) ? timings_[i] : 0.0f;
        if (blocks_[i]->isBypass()) {
            printf("  [%zu] %-30s  (Bypassed)\n", i, blocks_[i]->name());
        } else {
            printf("  [%zu] %-30s  %.3f ms\n", i, blocks_[i]->name(), t);
            total += t;
        }
    }
    printf("  %-34s  %.3f ms\n", "TOTAL", total);
    printf("============================\n\n");
}
