#include "blocks.h"
#include "isp_block.h"
#include <cstdio>
#include <memory>

// ============================================================================
// Demosaic — Bilinear Interpolation
//
// Converts single-channel Bayer (uint16_t) to 3-channel RGB (float in [0,1]).
// Supports all four Bayer patterns: RGGB, BGGR, GRBG, GBRG.
//
// Pattern is selected at compile time via the (RED_BX, RED_BY) template
// parameters, where (RED_BX, RED_BY) is the position of the R pixel within
// the 2x2 Bayer cell:
//
//   RGGB: (0,0)    BGGR: (1,1)    GRBG: (1,0)    GBRG: (0,1)
//
// The host-side dispatcher picks the right specialization based on the
// input FrameBuffer's PixelFormat. Each per-pixel branch (R / G-on-R-row /
// G-on-B-row / B) is resolved at compile time via `if constexpr`, so no
// runtime cost is paid for pattern flexibility.
// ============================================================================

// ----------------------------------------------------------------------------
// Per-pixel demosaic given already-fetched 3x3 neighbourhood.
// Inputs are floats so callers can pre-cast (saves repeated conversions when
// neighbours are reused across kernels). Returns (R, G, B) in a float3.
//
// Layout of the neighbourhood (centre = pixel being demosaiced):
//
//   nw  n  ne
//    w  c   e
//   sw  s  se
// ----------------------------------------------------------------------------
template <int RED_BX, int RED_BY>
__device__ __forceinline__ float3 demosaicPixel(
    int bx, int by, float c,
    float n, float s, float e, float w,
    float nw, float ne, float sw, float se)
{
    constexpr int BLUE_BX = 1 - RED_BX;
    constexpr int BLUE_BY = 1 - RED_BY;

    const bool is_red  = (bx == RED_BX)  && (by == RED_BY);
    const bool is_blue = (bx == BLUE_BX) && (by == BLUE_BY);
    const bool on_red_row = (by == RED_BY);

    float r, g, b;
    if (is_red) {
        // R pixel: G from cardinals, B from diagonals
        r = c;
        g = (n + s + e + w) * 0.25f;
        b = (nw + ne + sw + se) * 0.25f;
    } else if (is_blue) {
        // B pixel: G from cardinals, R from diagonals
        r = (nw + ne + sw + se) * 0.25f;
        g = (n + s + e + w) * 0.25f;
        b = c;
    } else if (on_red_row) {
        // G on red row: horizontal neighbours are R, vertical are B
        r = (e + w) * 0.5f;
        g = c;
        b = (n + s) * 0.5f;
    } else {
        // G on blue row: horizontal neighbours are B, vertical are R
        r = (n + s) * 0.5f;
        g = c;
        b = (e + w) * 0.5f;
    }
    return make_float3(r, g, b);
}

// ============================================================================
// Naive kernel — global-memory reads, boundary clamping
// ============================================================================
template <int RED_BX, int RED_BY>
__global__ void demosaicKernelNaive(const uint16_t* __restrict__ bayer,
                                    float* __restrict__ rgb,
                                    int width, int height,
                                    float norm_factor)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    auto fetch = [&] (int px, int py) -> float {
        px = min(max(px, 0), width  - 1);
        py = min(max(py, 0), height - 1);
        return static_cast<float>(bayer[py * width + px]);
    };

    const float c  = fetch(x,     y    );
    const float n  = fetch(x,     y - 1);
    const float s  = fetch(x,     y + 1);
    const float e  = fetch(x + 1, y    );
    const float w  = fetch(x - 1, y    );
    const float nw = fetch(x - 1, y - 1);
    const float ne = fetch(x + 1, y - 1);
    const float sw = fetch(x - 1, y + 1);
    const float se = fetch(x + 1, y + 1);

    const float3 rgb_v = demosaicPixel<RED_BX, RED_BY>(
        x & 1, y & 1, c, n, s, e, w, nw, ne, sw, se);

    const int out_idx = (y * width + x) * 3;
    rgb[out_idx + 0] = rgb_v.x * norm_factor;
    rgb[out_idx + 1] = rgb_v.y * norm_factor;
    rgb[out_idx + 2] = rgb_v.z * norm_factor;
}

// ============================================================================
// Optimized kernel — 16x16 tile + 1-pixel halo in shared memory
// ============================================================================
template <int RED_BX, int RED_BY>
__global__ void demosaicKernelOptimized(const uint16_t* __restrict__ bayer,
                                        float* __restrict__ rgb,
                                        int width, int height,
                                        float norm_factor)
{
    constexpr int TILE_W = 16;
    constexpr int TILE_H = 16;
    constexpr int HALO   = 1;
    constexpr int SMEM_W = TILE_W + 2 * HALO;  // 18
    constexpr int SMEM_H = TILE_H + 2 * HALO;  // 18

    __shared__ uint16_t smem[SMEM_H][SMEM_W];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx_g = blockIdx.x * TILE_W;
    const int by_g = blockIdx.y * TILE_H;

    // Cooperative load: 256 threads load 18*18 = 324 elements, with a
    // grid-stride loop so a few threads fetch a second element.
    const int tid = ty * blockDim.x + tx;
    for (int i = tid; i < SMEM_H * SMEM_W; i += blockDim.x * blockDim.y) {
        const int sy = i / SMEM_W;
        const int sx = i % SMEM_W;
        int gx = bx_g - HALO + sx;
        int gy = by_g - HALO + sy;
        gx = min(max(gx, 0), width  - 1);
        gy = min(max(gy, 0), height - 1);
        smem[sy][sx] = bayer[gy * width + gx];
    }
    __syncthreads();

    const int x = bx_g + tx;
    const int y = by_g + ty;
    if (x >= width || y >= height) return;

    const int sx = tx + HALO;
    const int sy = ty + HALO;

    auto sm = [&] (int dx, int dy) -> float {
        return static_cast<float>(smem[sy + dy][sx + dx]);
    };

    const float3 rgb_v = demosaicPixel<RED_BX, RED_BY>(
        x & 1, y & 1,
        sm( 0,  0),
        sm( 0, -1), sm( 0, +1), sm(+1,  0), sm(-1,  0),
        sm(-1, -1), sm(+1, -1), sm(-1, +1), sm(+1, +1));

    const int out_idx = (y * width + x) * 3;
    rgb[out_idx + 0] = rgb_v.x * norm_factor;
    rgb[out_idx + 1] = rgb_v.y * norm_factor;
    rgb[out_idx + 2] = rgb_v.z * norm_factor;
}

// ============================================================================
// ISPBlock implementations
// ============================================================================
class Demosaic : public ISPBlock {
public:
    explicit Demosaic(int bit_depth = 10)
        : norm_factor_(1.0f / static_cast<float>((1 << bit_depth) - 1)) {}

    const char* name() const override { return "Demosaic (Bilinear)"; }

    void process(const FrameBuffer& input, FrameBuffer& output,
                 cudaStream_t stream) override {
        if (!input.isBayer()) {
            fprintf(stderr, "[Demosaic] Error: expected Bayer input\n");
            return;
        }

        output.width    = input.width;
        output.height   = input.height;
        output.channels = 3;
        output.format   = PixelFormat::RGB_FLOAT;
        output.allocate();

        const dim3 block(16, 16);
        const dim3 grid((input.width  + block.x - 1) / block.x,
                        (input.height + block.y - 1) / block.y);
        const auto* d_in  = static_cast<const uint16_t*>(input.d_data);
        auto*       d_out = static_cast<float*>(output.d_data);
        const int   w     = input.width;
        const int   h     = input.height;
        const float nf    = norm_factor_;

        #define LAUNCH(RX, RY) \
            demosaicKernelNaive<RX, RY><<<grid, block, 0, stream>>>( \
                d_in, d_out, w, h, nf)

        switch (input.format) {
            case PixelFormat::BAYER_RGGB: LAUNCH(0, 0); break;
            case PixelFormat::BAYER_BGGR: LAUNCH(1, 1); break;
            case PixelFormat::BAYER_GRBG: LAUNCH(1, 0); break;
            case PixelFormat::BAYER_GBRG: LAUNCH(0, 1); break;
            default:
                fprintf(stderr, "[Demosaic] Error: unsupported Bayer format %d\n",
                        static_cast<int>(input.format));
        }
        #undef LAUNCH
    }

private:
    float norm_factor_;
};

class DemosaicOptimized : public ISPBlock {
public:
    explicit DemosaicOptimized(int bit_depth = 10)
        : norm_factor_(1.0f / static_cast<float>((1 << bit_depth) - 1)) {}

    const char* name() const override { return "Demosaic (Optimized)"; }

    void process(const FrameBuffer& input, FrameBuffer& output,
                 cudaStream_t stream) override {
        if (!input.isBayer()) {
            fprintf(stderr, "[Demosaic] Error: expected Bayer input\n");
            return;
        }

        output.width    = input.width;
        output.height   = input.height;
        output.channels = 3;
        output.format   = PixelFormat::RGB_FLOAT;
        output.allocate();

        const dim3 block(16, 16);
        const dim3 grid((input.width  + block.x - 1) / block.x,
                        (input.height + block.y - 1) / block.y);
        const auto* d_in  = static_cast<const uint16_t*>(input.d_data);
        auto*       d_out = static_cast<float*>(output.d_data);
        const int   w     = input.width;
        const int   h     = input.height;
        const float nf    = norm_factor_;

        #define LAUNCH(RX, RY) \
            demosaicKernelOptimized<RX, RY><<<grid, block, 0, stream>>>( \
                d_in, d_out, w, h, nf)

        switch (input.format) {
            case PixelFormat::BAYER_RGGB: LAUNCH(0, 0); break;
            case PixelFormat::BAYER_BGGR: LAUNCH(1, 1); break;
            case PixelFormat::BAYER_GRBG: LAUNCH(1, 0); break;
            case PixelFormat::BAYER_GBRG: LAUNCH(0, 1); break;
            default:
                fprintf(stderr, "[Demosaic] Error: unsupported Bayer format %d\n",
                        static_cast<int>(input.format));
        }
        #undef LAUNCH
    }

private:
    float norm_factor_;
};

// ============================================================================
// Factory functions
// ============================================================================
std::unique_ptr<ISPBlock> createDemosaic(int bit_depth) {
    return std::make_unique<Demosaic>(bit_depth);
}

std::unique_ptr<ISPBlock> createDemosaicOptimized(int bit_depth) {
    return std::make_unique<DemosaicOptimized>(bit_depth);
}
