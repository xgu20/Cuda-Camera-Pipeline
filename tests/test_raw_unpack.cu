#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "blocks.h"
#include "frame_buffer.h"
#include "isp_block.h"

// ============================================================================
// Host-side reference implementations — single source of truth for what the
// kernels should produce. Used to (a) build packed test input, and (b) verify
// kernel output against a known-good unpack.
// ============================================================================
namespace {

// Pack a width*height uint16 buffer (10-bit values in low bits) into MIPI
// CSI-2 RAW10 layout. width must be a multiple of 4.
std::vector<uint8_t> packMipi10(const std::vector<uint16_t>& src,
                                int width, int height) {
    const int groups_per_row = width / 4;
    const int packed_stride  = groups_per_row * 5;
    std::vector<uint8_t> out(static_cast<size_t>(packed_stride) * height);

    for (int y = 0; y < height; ++y) {
        const uint16_t* row_in = src.data() + y * width;
        uint8_t*        row_out = out.data() + y * packed_stride;
        for (int g = 0; g < groups_per_row; ++g) {
            const uint16_t p0 = row_in[g*4 + 0] & 0x3FF;
            const uint16_t p1 = row_in[g*4 + 1] & 0x3FF;
            const uint16_t p2 = row_in[g*4 + 2] & 0x3FF;
            const uint16_t p3 = row_in[g*4 + 3] & 0x3FF;
            row_out[g*5 + 0] = static_cast<uint8_t>(p0 >> 2);
            row_out[g*5 + 1] = static_cast<uint8_t>(p1 >> 2);
            row_out[g*5 + 2] = static_cast<uint8_t>(p2 >> 2);
            row_out[g*5 + 3] = static_cast<uint8_t>(p3 >> 2);
            row_out[g*5 + 4] = static_cast<uint8_t>(
                ((p3 & 0x3) << 6) | ((p2 & 0x3) << 4) |
                ((p1 & 0x3) << 2) | ((p0 & 0x3) << 0));
        }
    }
    return out;
}

// Build a FrameBuffer on the GPU from host-side packed bytes.
FrameBuffer makePackedFrame(const std::vector<uint8_t>& packed,
                            int width, int height, int bit_depth) {
    FrameBuffer fb;
    fb.width     = width;
    fb.height    = height;
    fb.channels  = 1;
    fb.format    = PixelFormat::BAYER_RGGB;
    fb.packing   = PixelPacking::PACKED_10_MIPI;
    fb.bit_depth = bit_depth;
    fb.allocate();
    if (fb.sizeBytes() != packed.size()) {
        std::cerr << "Size mismatch: fb=" << fb.sizeBytes()
                  << " packed=" << packed.size() << "\n";
        std::abort();
    }
    CUDA_CHECK(cudaMemcpy(fb.d_data, packed.data(), packed.size(),
                          cudaMemcpyHostToDevice));
    return fb;
}

// Download the unpacked output to host.
std::vector<uint16_t> downloadUnpacked(const FrameBuffer& fb) {
    std::vector<uint16_t> out(static_cast<size_t>(fb.width) * fb.height);
    CUDA_CHECK(cudaMemcpy(out.data(), fb.d_data,
                          out.size() * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    return out;
}

// Generate a deterministic 10-bit random Bayer image.
std::vector<uint16_t> makeRandomBayer(int width, int height, uint32_t seed) {
    std::vector<uint16_t> img(static_cast<size_t>(width) * height);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 1023);
    for (auto& p : img) p = static_cast<uint16_t>(dist(rng));
    return img;
}

// Time a callable that issues GPU work onto `stream`, averaged over `iters`.
// `setup` runs once outside the timing region.
template <typename Fn>
float timeMs(cudaStream_t stream, int iters, Fn&& fn) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));

    // Warm up
    fn();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(a, stream));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(b, stream));
    CUDA_CHECK(cudaEventSynchronize(b));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms / iters;
}

}  // namespace

// ============================================================================
// Test fixture
// ============================================================================
class RawUnpackTest : public ::testing::Test {
protected:
    void SetUp() override {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    void TearDown() override {
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }
    cudaStream_t stream_;
};

// ============================================================================
// Correctness: each variant must produce exactly the host-side reference.
// (Bit-exact equality, not approximate — unpack is integer arithmetic.)
// ============================================================================
TEST_F(RawUnpackTest, CorrectnessMatchesReference) {
    const int width = 128;
    const int height = 64;
    const int bit_depth = 10;

    const auto h_bayer = makeRandomBayer(width, height, /*seed=*/42);
    const auto h_packed = packMipi10(h_bayer, width, height);
    FrameBuffer fb_in = makePackedFrame(h_packed, width, height, bit_depth);

    struct Variant { const char* tag; std::unique_ptr<ISPBlock> block; };
    std::vector<Variant> variants;
    variants.push_back({"naive",       createRawUnpackNaive()});
    variants.push_back({"vec-store",   createRawUnpackVecStore()});
    variants.push_back({"vec-rw",      createRawUnpackVecRW()});
    variants.push_back({"vec-rw-grp4", createRawUnpackVecRWGrp4()});

    for (auto& v : variants) {
        FrameBuffer fb_out;
        v.block->process(fb_in, fb_out, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        const auto h_out = downloadUnpacked(fb_out);
        ASSERT_EQ(h_out.size(), h_bayer.size()) << "variant=" << v.tag;
        for (size_t i = 0; i < h_out.size(); ++i) {
            ASSERT_EQ(h_out[i], h_bayer[i])
                << "variant=" << v.tag << " idx=" << i;
        }

        fb_out.free();
    }

    fb_in.free();
}

// ============================================================================
// Cross-validation: all three variants produce identical output (bit-for-bit).
// Catches "all variants are wrong in the same way" if the reference test fails
// to spot something, and also catches variants drifting apart over time.
// ============================================================================
TEST_F(RawUnpackTest, AllVariantsAgree) {
    const int width = 1920;
    const int height = 1080;
    const int bit_depth = 10;

    const auto h_bayer = makeRandomBayer(width, height, /*seed=*/7);
    const auto h_packed = packMipi10(h_bayer, width, height);
    FrameBuffer fb_in = makePackedFrame(h_packed, width, height, bit_depth);

    auto run = [&](std::unique_ptr<ISPBlock>& block) {
        FrameBuffer fb_out;
        block->process(fb_in, fb_out, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        auto h = downloadUnpacked(fb_out);
        fb_out.free();
        return h;
    };

    auto naive_block        = createRawUnpackNaive();
    auto vecstore_block     = createRawUnpackVecStore();
    auto vecrw_block        = createRawUnpackVecRW();
    auto vecrwgrp4_block    = createRawUnpackVecRWGrp4();

    const auto h_naive     = run(naive_block);
    const auto h_vecstore  = run(vecstore_block);
    const auto h_vecrw     = run(vecrw_block);
    const auto h_vecrwgrp4 = run(vecrwgrp4_block);

    EXPECT_EQ(h_naive, h_vecstore)  << "vec-store diverges from naive";
    EXPECT_EQ(h_naive, h_vecrw)     << "vec-rw diverges from naive";
    EXPECT_EQ(h_naive, h_vecrwgrp4) << "vec-rw-grp4 diverges from naive";

    fb_in.free();
}

// ============================================================================
// Performance: head-to-head benchmark of the three variants on a 4K-ish frame.
// Prints ms / GB/s / speedup so you can read it off without external tooling.
// ============================================================================
TEST_F(RawUnpackTest, Performance_4K) {
    const int width = 3840;
    const int height = 2160;
    const int bit_depth = 10;
    const int iters = 200;

    // Random input so the compiler can't constant-fold anything.
    const auto h_bayer = makeRandomBayer(width, height, /*seed=*/123);
    const auto h_packed = packMipi10(h_bayer, width, height);
    FrameBuffer fb_in = makePackedFrame(h_packed, width, height, bit_depth);

    struct Variant {
        const char* tag;
        std::unique_ptr<ISPBlock> block;
        FrameBuffer fb_out{};
        float ms = 0.0f;
    };
    std::vector<Variant> variants;
    variants.push_back({"naive",       createRawUnpackNaive()});
    variants.push_back({"vec-store",   createRawUnpackVecStore()});
    variants.push_back({"vec-rw",      createRawUnpackVecRW()});
    variants.push_back({"vec-rw-grp4", createRawUnpackVecRWGrp4()});

    // We benchmark only the kernel cost — output buffer is allocated once
    // before timing so cudaMalloc isn't part of the measurement.
    for (auto& v : variants) {
        v.block->process(fb_in, v.fb_out, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        v.ms = timeMs(stream_, iters, [&] {
            FrameBuffer reuse = v.fb_out;     // shallow copy: same d_data
            v.block->process(fb_in, reuse, stream_);
            // process() sees reuse.d_data != nullptr and skips allocate(),
            // so the timed region only contains the kernel launch.
        });
    }

    // Memory traffic per frame: read packed + write unpacked
    const size_t packed_bytes  = static_cast<size_t>(width / 4) * 5 * height;
    const size_t unpacked_bytes = static_cast<size_t>(width) * height * sizeof(uint16_t);
    const size_t total_bytes = packed_bytes + unpacked_bytes;

    std::cout << "[ PERF     ] Resolution: " << width << "x" << height
              << " (" << iters << " iters)\n";
    const float ms_naive = variants[0].ms;
    for (const auto& v : variants) {
        const float gbps = (total_bytes / (v.ms * 1e-3f)) / 1.0e9f;
        const float speedup = ms_naive / v.ms;
        std::cout << "[ PERF     ] " << v.tag
                  << "  avg=" << v.ms << " ms"
                  << "  bw="  << gbps << " GB/s"
                  << "  speedup=" << speedup << "x\n";
    }

    for (auto& v : variants) v.fb_out.free();
    fb_in.free();
}

// ============================================================================
// 8-bit promotion: UNPACKED_U8 input must be zero-extended to UNPACKED_U16.
// ============================================================================
TEST_F(RawUnpackTest, PromotesU8ToU16) {
    const int width = 256;
    const int height = 64;
    const int bit_depth = 8;

    // Deterministic random uint8 input.
    std::vector<uint8_t> h_in(static_cast<size_t>(width) * height);
    std::mt19937 rng(2024);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& b : h_in) b = static_cast<uint8_t>(dist(rng));

    FrameBuffer fb_in;
    fb_in.width     = width;
    fb_in.height    = height;
    fb_in.channels  = 1;
    fb_in.format    = PixelFormat::BAYER_RGGB;
    fb_in.packing   = PixelPacking::UNPACKED_U8;
    fb_in.bit_depth = bit_depth;
    fb_in.allocate();
    ASSERT_EQ(fb_in.sizeBytes(), h_in.size());
    CUDA_CHECK(cudaMemcpy(fb_in.d_data, h_in.data(), h_in.size(),
                          cudaMemcpyHostToDevice));

    auto block = createRawUnpack();
    FrameBuffer fb_out;
    block->process(fb_in, fb_out, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Output should be UNPACKED_U16, same dimensions, with values byte-extended.
    ASSERT_NE(fb_out.d_data, nullptr);
    EXPECT_EQ(fb_out.packing, PixelPacking::UNPACKED_U16);
    EXPECT_EQ(fb_out.width,  width);
    EXPECT_EQ(fb_out.height, height);

    const auto h_out = downloadUnpacked(fb_out);
    ASSERT_EQ(h_out.size(), h_in.size());
    for (size_t i = 0; i < h_in.size(); ++i) {
        ASSERT_EQ(h_out[i], static_cast<uint16_t>(h_in[i])) << "idx=" << i;
    }

    fb_out.free();
    fb_in.free();
}

// ============================================================================
// Rejection: MIPI10 with width not a multiple of 4 should fail cleanly.
// ============================================================================
TEST_F(RawUnpackTest, RejectsBadWidth) {
    FrameBuffer fb_in;
    fb_in.width     = 1022;   // not a multiple of 4
    fb_in.height    = 4;
    fb_in.channels  = 1;
    fb_in.format    = PixelFormat::BAYER_RGGB;
    fb_in.packing   = PixelPacking::PACKED_10_MIPI;
    fb_in.bit_depth = 10;
    fb_in.allocate();

    auto block = createRawUnpackVecStore();
    FrameBuffer fb_out;
    EXPECT_THROW(block->process(fb_in, fb_out, stream_), std::invalid_argument);

    fb_in.free();
}
