#include "blocks.h"
#include "isp_block.h"
#include <cstdio>
#include <memory>

// ============================================================================
// RawUnpack — convert packed sensor data into UNPACKED_U16 Bayer.
//
// Three RAW10 unpack kernels coexist for benchmarking. Pick a variant via:
//   createRawUnpackNaive()     — scalar loads, 4 separate uint16 stores
//   createRawUnpackVecStore()  — scalar loads, 1 ushort4 store
//   createRawUnpackVecRW()     — vector loads + ushort4 stores, N groups/thread
//
// Subsequent ISP blocks (BLC / Demosaic / etc.) only ever see UNPACKED_U16
// data, so no other block needs to know about packing.
// ============================================================================

// ----------------------------------------------------------------------------
// MIPI CSI-2 RAW10 layout (per 5-byte group, 4 pixels):
//
//   byte 0 = P0[9:2]
//   byte 1 = P1[9:2]
//   byte 2 = P2[9:2]
//   byte 3 = P3[9:2]
//   byte 4 = (P3[1:0]<<6) | (P2[1:0]<<4) | (P1[1:0]<<2) | P0[1:0]
//
// Output is uint16_t per pixel with the 10-bit value stored in the low bits
// (i.e. the high 6 bits of the uint16_t are zero).
//
// Common kernel parameters:
//   packed        — device pointer to packed bytes, row-major
//   packed_stride — bytes per row in `packed`  (= ((width+3)/4)*5 typically)
//   unpacked      — device pointer to uint16_t Bayer output (width*height*2 B)
//   width, height — image dimensions in pixels
//
// Precondition (enforced by RawUnpack::process on the host side):
//   width % 4 == 0
// The vector store paths write 4 pixels at a time and cannot handle a
// partial trailing group. If you ever want arbitrary widths, relax the
// host-side check and add explicit tail handling per kernel.
// ----------------------------------------------------------------------------

// ============================================================================
// Shared device helper: unpack one 5-byte group → 4 uint16_t pixels.
// Used by the scalar/vec-store kernels; the vec-rw kernel may inline its own
// vectorized load if it wants to issue uchar4 / uint32 reads instead.
// ============================================================================
__device__ __forceinline__ ushort4 unpackOneGroup(const uint8_t* row, int gx) {
    const uint8_t b0 = row[gx * 5 + 0];
    const uint8_t b1 = row[gx * 5 + 1];
    const uint8_t b2 = row[gx * 5 + 2];
    const uint8_t b3 = row[gx * 5 + 3];
    const uint8_t lo = row[gx * 5 + 4];

    return make_ushort4(
        (uint16_t(b0) << 2) | ((lo >> 0) & 0x3),
        (uint16_t(b1) << 2) | ((lo >> 2) & 0x3),
        (uint16_t(b2) << 2) | ((lo >> 4) & 0x3),
        (uint16_t(b3) << 2) | ((lo >> 6) & 0x3));
}

// ============================================================================
// Variant 1: Naive — scalar byte loads, 4 separate uint16 stores
// ============================================================================
__global__ void unpackMipi10NaiveKernel(const uint8_t* __restrict__ packed,
                                        size_t packed_stride,
                                        uint16_t* __restrict__ unpacked,
                                        int width, int height) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx * 4 >= width || y >= height) return;

    const uint8_t* row = packed + packed_stride * y;
    const ushort4 px = unpackOneGroup(row, gx);

    uint16_t* dst = &unpacked[y * width + gx * 4];
    dst[0] = px.x;
    dst[1] = px.y;
    dst[2] = px.z;
    dst[3] = px.w;
}

// ============================================================================
// Variant 2: VecStore — scalar byte loads, 1 ushort4 store
// ============================================================================
__global__ void unpackMipi10VecStoreKernel(const uint8_t* __restrict__ packed,
                                           size_t packed_stride,
                                           uint16_t* __restrict__ unpacked,
                                           int width, int height) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int y  = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx * 4 >= width || y >= height) return;

    const uint8_t* row = packed + packed_stride * y;
    const ushort4 px = unpackOneGroup(row, gx);

    *reinterpret_cast<ushort4*>(&unpacked[y * width + gx * 4]) = px;
}

// ============================================================================
// Variant 3: VecRW — vectorized reads + ushort4 stores, N groups per thread
//
// Goal: reduce launch overhead and improve memory throughput by having each
// thread handle multiple groups, and (optionally) by issuing wider loads
// (uchar4, uint32_t) for the packed bytes.
//
// Two design knobs you can play with:
//   1. GROUPS_PER_THREAD (compile-time): 2, 4, 8...
//   2. The load pattern inside the loop — scalar bytes, uchar4, uint32_t, etc.
//
// The launcher below already passes a smaller `grid.x` reflecting that each
// thread covers GROUPS_PER_THREAD groups. See launchVecRW().
// ============================================================================
constexpr int VECRW_GROUPS_PER_THREAD = 2;  // tweak and re-bench

__global__ void unpackMipi10VecRWKernel(const uint8_t* __restrict__ packed,
                                        size_t packed_stride,
                                        uint16_t* __restrict__ unpacked,
                                        int width, int height) {
    // TODO(you): implement the vectorized read + write version.
    //
    // Suggested structure:
    //
    //   const int tx       = blockIdx.x * blockDim.x + threadIdx.x;
    //   const int gx_base  = tx * VECRW_GROUPS_PER_THREAD;
    //   const int y        = blockIdx.y * blockDim.y + threadIdx.y;
    //   if (gx_base * 4 >= width || y >= height) return;
    //
    //   const uint8_t* row = packed + packed_stride * y;
    //
    //   #pragma unroll
    //   for (int i = 0; i < VECRW_GROUPS_PER_THREAD; ++i) {
    //       const int gx = gx_base + i;
    //       if (gx * 4 >= width) break;           // ragged tail of row
    //
    //       // Option A — reuse the shared helper (scalar 5 byte loads):
    //       ushort4 px = unpackOneGroup(row, gx);
    //
    //       // Option B — vectorized load. uchar4 wants 4-byte alignment;
    //       // row + gx*5 is generally NOT 4-aligned, so be careful.
    //       //   - Easiest: use __ldg for byte loads to hit the read-only cache.
    //       //   - Aligned trick: have each thread process 4 groups (= 20 bytes)
    //       //     and read 5 × uint32_t which IS aligned. Requires reorganizing
    //       //     the loop to share state across groups within a thread.
    //
    //       *reinterpret_cast<ushort4*>(&unpacked[y * width + gx * 4]) = px;
    //   }
    //
    // Once correct, benchmark vs VecStore. If you don't see improvement, try
    // bumping VECRW_GROUPS_PER_THREAD or changing the block shape in launchVecRW.
    const int tx      = blockIdx.x * blockDim.x + threadIdx.x;
    const int gx_base = tx * VECRW_GROUPS_PER_THREAD;
    const int y       = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx_base * 4 >= width || y >= height) return;

    const uint8_t* row = packed + packed_stride * y;

    #pragma unroll
    for (int i = 0; i < VECRW_GROUPS_PER_THREAD; ++i) {
        const int gx = gx_base + i;
        if (gx * 4 >= width) break;

        // Scheme A: 5 scalar __ldg byte loads through the read-only cache.
        // We deliberately avoid __ldg(uchar4*) here — row + gx*5 is only
        // 4B-aligned when gx % 4 == 0, so the vector load would fault.
        const uint8_t b0 = __ldg(&row[gx * 5 + 0]);
        const uint8_t b1 = __ldg(&row[gx * 5 + 1]);
        const uint8_t b2 = __ldg(&row[gx * 5 + 2]);
        const uint8_t b3 = __ldg(&row[gx * 5 + 3]);
        const uint8_t lo = __ldg(&row[gx * 5 + 4]);

        const ushort4 px = make_ushort4(
            (uint16_t(b0) << 2) | ((lo >> 0) & 0x3),
            (uint16_t(b1) << 2) | ((lo >> 2) & 0x3),
            (uint16_t(b2) << 2) | ((lo >> 4) & 0x3),
            (uint16_t(b3) << 2) | ((lo >> 6) & 0x3));

        *reinterpret_cast<ushort4*>(&unpacked[y * width + gx * 4]) = px;
    }

}

// ============================================================================
// Variant 4: VecRWGrp4 — 5×uint32 aligned reads per thread, 4 ushort4 writes
//
// Each thread covers 4 consecutive groups (= 20 bytes = 5 uint32_t). Because
// 20 is a multiple of 4 and gx_base = tx * 4 (so gx_base*5 is a multiple of
// 20), the 5 uint32 loads are guaranteed 4B-aligned and legal.
//
// Bytes 0..19 of the 20-byte block map to groups as:
//   w0 = bytes  0.. 3  -> g0.b0 g0.b1 g0.b2 g0.b3
//   w1 = bytes  4.. 7  -> g0.lo g1.b0 g1.b1 g1.b2
//   w2 = bytes  8..11  -> g1.b3 g1.lo g2.b0 g2.b1
//   w3 = bytes 12..15  -> g2.b2 g2.b3 g2.lo g3.b0
//   w4 = bytes 16..19  -> g3.b1 g3.b2 g3.b3 g3.lo
//
// Precondition (enforced host-side in process()):
//   width % 16 == 0, i.e. groups_per_row % 4 == 0
// This lets us avoid any per-thread tail handling.
// ============================================================================
constexpr int VECRWGRP4_GROUPS_PER_THREAD = 4;

__device__ __forceinline__ uint8_t byteOf(uint32_t w, int k) {
    return static_cast<uint8_t>(w >> (k * 8));
}

__device__ __forceinline__ ushort4 packGroup(uint8_t b0, uint8_t b1,
                                             uint8_t b2, uint8_t b3,
                                             uint8_t lo) {
    return make_ushort4(
        (uint16_t(b0) << 2) | ((lo >> 0) & 0x3),
        (uint16_t(b1) << 2) | ((lo >> 2) & 0x3),
        (uint16_t(b2) << 2) | ((lo >> 4) & 0x3),
        (uint16_t(b3) << 2) | ((lo >> 6) & 0x3));
}

__global__ void unpackMipi10VecRWGrp4Kernel(const uint8_t* __restrict__ packed,
                                            size_t packed_stride,
                                            uint16_t* __restrict__ unpacked,
                                            int width, int height) {
    const int tx      = blockIdx.x * blockDim.x + threadIdx.x;
    const int gx_base = tx * VECRWGRP4_GROUPS_PER_THREAD;
    const int y       = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx_base * 4 >= width || y >= height) return;

    const uint8_t*  row   = packed + packed_stride * y;
    const uint32_t* row32 = reinterpret_cast<const uint32_t*>(row + gx_base * 5);

    const uint32_t w0 = __ldg(&row32[0]);
    const uint32_t w1 = __ldg(&row32[1]);
    const uint32_t w2 = __ldg(&row32[2]);
    const uint32_t w3 = __ldg(&row32[3]);
    const uint32_t w4 = __ldg(&row32[4]);

    const ushort4 g0 = packGroup(byteOf(w0,0), byteOf(w0,1), byteOf(w0,2), byteOf(w0,3),
                                 byteOf(w1,0));
    const ushort4 g1 = packGroup(byteOf(w1,1), byteOf(w1,2), byteOf(w1,3), byteOf(w2,0),
                                 byteOf(w2,1));
    const ushort4 g2 = packGroup(byteOf(w2,2), byteOf(w2,3), byteOf(w3,0), byteOf(w3,1),
                                 byteOf(w3,2));
    const ushort4 g3 = packGroup(byteOf(w3,3), byteOf(w4,0), byteOf(w4,1), byteOf(w4,2),
                                 byteOf(w4,3));

    ushort4* out4 = reinterpret_cast<ushort4*>(&unpacked[y * width + gx_base * 4]);
    out4[0] = g0;
    out4[1] = g1;
    out4[2] = g2;
    out4[3] = g3;
}


// ============================================================================
// 8-bit promotion — UNPACKED_U8 → UNPACKED_U16 (zero-extend each byte).
//
// Scalar one-pixel-per-thread kernel. The 8-bit data is small (1 B/pixel,
// e.g. 4 MB for 2592x1536) and this is bandwidth-bound on the writes
// (2 B/pixel out > 1 B/pixel in), so a vectorized variant would buy at
// most a couple of percent and isn't worth the code. Bit-depth is *not*
// rescaled — Demosaic's normalisation already divides by (2^bit_depth - 1),
// so 8-bit input goes through naturally as long as the sidecar's
// `bit_depth` is 8.
// ============================================================================
__global__ void promoteU8ToU16Kernel(const uint8_t* __restrict__ in_u8,
                                     uint16_t* __restrict__ out_u16,
                                     int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int idx = y * width + x;
    out_u16[idx] = static_cast<uint16_t>(in_u8[idx]);
}

// ============================================================================
// Per-variant launchers — each owns its own grid/block geometry
// ============================================================================
namespace {

void launchPromoteU8(const FrameBuffer& in, FrameBuffer& out, cudaStream_t s) {
    const dim3 block(32, 8);
    const dim3 grid((in.width  + block.x - 1) / block.x,
                    (in.height + block.y - 1) / block.y);
    promoteU8ToU16Kernel<<<grid, block, 0, s>>>(
        static_cast<const uint8_t*>(in.d_data),
        static_cast<uint16_t*>(out.d_data),
        in.width, in.height);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[RawUnpack/PromoteU8] launch error: %s\n",
                cudaGetErrorString(e));
    }
}

void launchNaive(const FrameBuffer& in, FrameBuffer& out, cudaStream_t s) {
    const int groups_per_row = (in.width + 3) / 4;
    const dim3 block(64, 4);
    const dim3 grid((groups_per_row + block.x - 1) / block.x,
                    (in.height      + block.y - 1) / block.y);
    unpackMipi10NaiveKernel<<<grid, block, 0, s>>>(
        static_cast<const uint8_t*>(in.d_data), in.stride,
        static_cast<uint16_t*>(out.d_data),
        in.width, in.height);
}

void launchVecStore(const FrameBuffer& in, FrameBuffer& out, cudaStream_t s) {
    const int groups_per_row = (in.width + 3) / 4;
    const dim3 block(64, 4);
    const dim3 grid((groups_per_row + block.x - 1) / block.x,
                    (in.height      + block.y - 1) / block.y);
    unpackMipi10VecStoreKernel<<<grid, block, 0, s>>>(
        static_cast<const uint8_t*>(in.d_data), in.stride,
        static_cast<uint16_t*>(out.d_data),
        in.width, in.height);
}

void launchVecRW(const FrameBuffer& in, FrameBuffer& out, cudaStream_t s) {
    const int groups_per_row = (in.width + 3) / 4;
    const int threads_x = (groups_per_row + VECRW_GROUPS_PER_THREAD - 1) /
                          VECRW_GROUPS_PER_THREAD;
    // Block shape is independent of the other variants — tweak for occupancy.
    const dim3 block(32, 8);
    const dim3 grid((threads_x  + block.x - 1) / block.x,
                    (in.height  + block.y - 1) / block.y);
    unpackMipi10VecRWKernel<<<grid, block, 0, s>>>(
        static_cast<const uint8_t*>(in.d_data), in.stride,
        static_cast<uint16_t*>(out.d_data),
        in.width, in.height);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[RawUnpack/VecRW] launch error: %s\n",
                cudaGetErrorString(e));
    }
}

void launchVecRWGrp4(const FrameBuffer& in, FrameBuffer& out, cudaStream_t s) {
    // Precondition (checked in process()): width % 16 == 0, so groups_per_row
    // is a multiple of VECRWGRP4_GROUPS_PER_THREAD and no tail handling needed.
    const int groups_per_row = in.width / 4;
    const int threads_x      = groups_per_row / VECRWGRP4_GROUPS_PER_THREAD;
    const dim3 block(32, 8);
    const dim3 grid((threads_x + block.x - 1) / block.x,
                    (in.height + block.y - 1) / block.y);
    unpackMipi10VecRWGrp4Kernel<<<grid, block, 0, s>>>(
        static_cast<const uint8_t*>(in.d_data), in.stride,
        static_cast<uint16_t*>(out.d_data),
        in.width, in.height);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[RawUnpack/VecRWGrp4] launch error: %s\n",
                cudaGetErrorString(e));
    }
}

}  // namespace

// ============================================================================
// ISPBlock implementation
// ============================================================================
enum class UnpackVariant { Naive, VecStore, VecRW, VecRWGrp4 };

class RawUnpack : public ISPBlock {
public:
    explicit RawUnpack(UnpackVariant variant) : variant_(variant) {}

    const char* name() const override {
        switch (variant_) {
            case UnpackVariant::Naive:     return "RawUnpack (naive)";
            case UnpackVariant::VecStore:  return "RawUnpack (vec-store)";
            case UnpackVariant::VecRW:     return "RawUnpack (vec-rw)";
            case UnpackVariant::VecRWGrp4: return "RawUnpack (vec-rw-grp4)";
        }
        return "RawUnpack";
    }

    void process(const FrameBuffer& input, FrameBuffer& output,
                 cudaStream_t stream) override {
        if (!input.isBayer()) {
            fprintf(stderr, "[RawUnpack] Error: expected Bayer input\n");
            return;
        }

        // Fast path: already unpacked uint16 → just alias, no work.
        if (input.packing == PixelPacking::UNPACKED_U16) {
            output = input;
            return;
        }

        // 8-bit input → allocate a uint16 output and promote (zero-extend).
        if (input.packing == PixelPacking::UNPACKED_U8) {
            output.width     = input.width;
            output.height    = input.height;
            output.channels  = 1;
            output.format    = input.format;
            output.packing   = PixelPacking::UNPACKED_U16;
            output.bit_depth = input.bit_depth;
            output.allocate();
            launchPromoteU8(input, output, stream);
            return;
        }

        if (input.packing != PixelPacking::PACKED_10_MIPI) {
            fprintf(stderr, "[RawUnpack] Error: unsupported packing %d\n",
                    static_cast<int>(input.packing));
            return;
        }

        // Vector store paths can't handle a partial trailing group.
        // SensorConfig already rejects this at load time, but check here too
        // so callers that build a FrameBuffer manually (e.g. tests) get a
        // clean error instead of an out-of-bounds write.
        if (input.width % 4 != 0) {
            fprintf(stderr,
                    "[RawUnpack] Error: MIPI10 unpack requires width %% 4 == 0 "
                    "(got width=%d)\n", input.width);
            return;
        }

        // VecRWGrp4 processes 4 groups (= 16 pixels) per thread without any
        // tail handling, so it needs groups_per_row to be a multiple of 4.
        if (variant_ == UnpackVariant::VecRWGrp4 && input.width % 16 != 0) {
            fprintf(stderr,
                    "[RawUnpack/VecRWGrp4] Error: requires width %% 16 == 0 "
                    "(got width=%d). Use VecStore / VecRW for arbitrary widths.\n",
                    input.width);
            return;
        }

        // Allocate the unpacked uint16 output (all variants share this).
        output.width     = input.width;
        output.height    = input.height;
        output.channels  = 1;
        output.format    = input.format;
        output.packing   = PixelPacking::UNPACKED_U16;
        output.bit_depth = input.bit_depth;
        output.allocate();

        // Dispatch to the variant-specific launcher.
        switch (variant_) {
            case UnpackVariant::Naive:     launchNaive(input, output, stream);     break;
            case UnpackVariant::VecStore:  launchVecStore(input, output, stream);  break;
            case UnpackVariant::VecRW:     launchVecRW(input, output, stream);     break;
            case UnpackVariant::VecRWGrp4: launchVecRWGrp4(input, output, stream); break;
        }
    }

private:
    UnpackVariant variant_;
};

// ============================================================================
// Factory functions — default selects the best fully-implemented variant.
// ============================================================================
std::unique_ptr<ISPBlock> createRawUnpack() {
    return std::make_unique<RawUnpack>(UnpackVariant::VecStore); // Fastest
}
std::unique_ptr<ISPBlock> createRawUnpackNaive() {
    return std::make_unique<RawUnpack>(UnpackVariant::Naive);
}
std::unique_ptr<ISPBlock> createRawUnpackVecStore() {
    return std::make_unique<RawUnpack>(UnpackVariant::VecStore);
}
std::unique_ptr<ISPBlock> createRawUnpackVecRW() {
    return std::make_unique<RawUnpack>(UnpackVariant::VecRW);
}
std::unique_ptr<ISPBlock> createRawUnpackVecRWGrp4() {
    return std::make_unique<RawUnpack>(UnpackVariant::VecRWGrp4);
}
