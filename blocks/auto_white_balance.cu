#include "blocks.h"
#include "isp_block.h"
#include <cstdio>
#include <memory>
#include <stdexcept>

// Apply per-channel white-balance gains. `gains` is a device array of 4 floats
// laid out as [r, gr, gb, b]. Keeping the gains in device memory lets the
// Gray World path compute them on the GPU without a host round-trip.
template <int RED_BX, int RED_BY>
__global__ void applyWhiteBalanceKernel(uint16_t* data, int width, int height,
                                        uint16_t cut_off, const float* __restrict__ gains) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    constexpr int BLUE_BX = 1 - RED_BX;
    constexpr int BLUE_BY = 1 - RED_BY;
    bool is_red= ((x & 1) == RED_BX) && ((y & 1) == RED_BY);
    bool is_blue = ((x & 1) == BLUE_BX) && ((y & 1) == BLUE_BY);
    bool on_red_row = ((y & 1) == RED_BY);

    float gain;
    if (is_red)          gain = gains[0];  // r
    else if (is_blue)    gain = gains[3];  // b
    else if (on_red_row) gain = gains[1];  // gr
    else                 gain = gains[2];  // gb

    float px = gain * static_cast<float>(data[y * width + x]);
    data[y * width + x] = px < static_cast<float>(cut_off) ?  px : cut_off;
}

// Gray World statistics: accumulate per-channel pixel sums and counts over the
// whole Bayer frame. Layout of the output arrays (index): 0=R, 1=Gr, 2=Gb, 3=B.
//
// NOTE: this is the naive version — every thread atomicAdds directly to the 4
// global accumulators. Correct but contended. Later we can replace it with a
// shared-memory (or warp-shuffle) block reduction + one atomicAdd per block.
template <int RED_BX, int RED_BY>
__global__ void awbGrayWorldStatistics(const uint16_t* data, int width, int height,
                                       unsigned long long* sums) {
    __shared__ unsigned long long s_sums[4];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < 4) {
        s_sums[tid] = 0;
    }
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x<width && y<height) {
        constexpr int BLUE_BX = 1 - RED_BX;
        constexpr int BLUE_BY = 1 - RED_BY;
        const bool is_red     = ((x & 1) == RED_BX) && ((y & 1) == RED_BY);
        const bool is_blue    = ((x & 1) == BLUE_BX) && ((y & 1) == BLUE_BY);
        const bool on_red_row = ((y & 1) == RED_BY);
    
        int ch;
        if (is_red)          ch = 0;  // R
        else if (is_blue)    ch = 3;  // B
        else if (on_red_row) ch = 1;  // Gr (green on the red row)
        else                 ch = 2;  // Gb (green on the blue row)
    
        const unsigned long long px = static_cast<unsigned long long>(data[y * width + x]);
    
        atomicAdd(&s_sums[ch], px);
        
    }

    __syncthreads();

    if (tid < 4) {
        atomicAdd(&sums[tid], s_sums[tid]);
    }

}

// Compute Gray World gains entirely on the device from the per-channel sums,
// writing [r, gr, gb, b] into `gains`. Launched with a single thread — the work
// is trivial and this keeps the gains in device memory (no host round-trip).
__global__ void awbComputeGains(const unsigned long long* sums,
                                unsigned long long per_channel, float* gains) {
    auto avg = [per_channel](unsigned long long s) -> double {
        return per_channel ? static_cast<double>(s) / static_cast<double>(per_channel) : 0.0;
    };
    const double avgR  = avg(sums[0]);
    const double avgGr = avg(sums[1]);
    const double avgGb = avg(sums[2]);
    const double avgB  = avg(sums[3]);
    const double avgG  = 0.5 * (avgGr + avgGb);  // green is the reference

    gains[0] = (avgR > 0.0) ? static_cast<float>(avgG / avgR) : 1.0f;  // r
    gains[1] = 1.0f;                                                   // gr
    gains[2] = 1.0f;                                                   // gb
    gains[3] = (avgB > 0.0) ? static_cast<float>(avgG / avgB) : 1.0f;  // b
}

enum class AwbAlgorithm {
    Manual,
    GrayWorld,
};

class AutoWhiteBalance : public ISPBlock {
public:
    AutoWhiteBalance(AwbAlgorithm algo, WhiteBalanceGains gains, int bit_depth,
                     uint16_t cut_off)
        : algo_(algo), gains_(gains),
          cut_off_(makeCutOff(bit_depth, cut_off)) {}

    ~AutoWhiteBalance() override {
        if (d_sums_)  cudaFree(d_sums_);
        if (d_gains_) cudaFree(d_gains_);
    }

    const char *name() const override {
        switch (algo_) {
            case AwbAlgorithm::Manual:
                return "ManualWhiteBalance";
            
            case AwbAlgorithm::GrayWorld:
                return "AutoWhiteBalance - GrayWorld";
        } 

        // Default
        return "AutoWhiteBalance";
    }

    void process(const FrameBuffer &input, FrameBuffer &output,
            cudaStream_t stream) override {
        if (!input.isBayer() || input.packing != PixelPacking::UNPACKED_U16) {
            throw std::invalid_argument(
                "AutoWhiteBalance requires unpacked uint16 Bayer input");
        }
        if ((input.width & 1) != 0 || (input.height & 1) != 0) {
            throw std::invalid_argument(
                "AutoWhiteBalance requires even Bayer dimensions");
        }

        output = input;
        auto* d_out = static_cast<uint16_t*>(output.d_data);
        const int   w     = input.width;
        const int   h     = input.height;

        const dim3 block(16, 16);
        const dim3 grid((input.width  + block.x - 1) / block.x,
                        (input.height + block.y - 1) / block.y);

        // Persistent device gains buffer [r, gr, gb, b], allocated once.
        if (!d_gains_) CUDA_CHECK(cudaMalloc(&d_gains_, 4 * sizeof(float)));

        // Apply the device-resident gains with the Bayer phase (RX, RY).
        #define LAUNCH_APPLY(RX, RY) \
            applyWhiteBalanceKernel<RX, RY><<<grid, block, 0, stream>>>(d_out, w, h, cut_off_, d_gains_)

        // Run the Gray World statistics kernel with the Bayer phase (RX, RY).
        #define LAUNCH_STATS(RX, RY) \
            awbGrayWorldStatistics<RX, RY><<<grid, block, 0, stream>>>(d_out, w, h, d_sums_)

        // Dispatch a per-format STMT(RX, RY) statement based on the Bayer layout.
        #define DISPATCH_FORMAT(STMT)                                                    \
            switch (input.format) {                                                      \
                case PixelFormat::BAYER_RGGB: STMT(0, 0); break;                         \
                case PixelFormat::BAYER_BGGR: STMT(1, 1); break;                         \
                case PixelFormat::BAYER_GRBG: STMT(1, 0); break;                         \
                case PixelFormat::BAYER_GBRG: STMT(0, 1); break;                         \
                default: break;                                                          \
            }

        if (algo_ == AwbAlgorithm::GrayWorld) {
            // Phase 1: per-channel pixel sums (d_sums_ reused across frames).
            if (!d_sums_) CUDA_CHECK(cudaMalloc(&d_sums_, 4 * sizeof(unsigned long long)));
            CUDA_CHECK(cudaMemsetAsync(d_sums_, 0, 4 * sizeof(unsigned long long), stream));
            DISPATCH_FORMAT(LAUNCH_STATS);

            // Phase 2: compute gains on the device — no host round-trip / sync.
            // Each channel has one slot per 2x2 Bayer tile => (w*h)/4 pixels.
            const unsigned long long perChannel =
                static_cast<unsigned long long>(w) * static_cast<unsigned long long>(h) / 4ULL;
            awbComputeGains<<<1, 1, 0, stream>>>(d_sums_, perChannel, d_gains_);
        } else {  // Manual: upload the fixed gains to the device once.
            if (!manual_uploaded_) {
                const float hg[4] = {gains_.r, gains_.gr, gains_.gb, gains_.b};
                CUDA_CHECK(cudaMemcpy(d_gains_, hg, sizeof(hg), cudaMemcpyHostToDevice));
                manual_uploaded_ = true;
            }
        }

        // Phase 3: apply gains (common to both algorithms).
        DISPATCH_FORMAT(LAUNCH_APPLY);

        #undef DISPATCH_FORMAT
        #undef LAUNCH_STATS
        #undef LAUNCH_APPLY

    }

private:
    static uint16_t makeCutOff(int bit_depth, uint16_t cut_off) {
        if (bit_depth < 1 || bit_depth > 16) {
            throw std::invalid_argument(
                "AutoWhiteBalance bit_depth must be in [1, 16]");
        }
        const uint32_t max_code = (1u << bit_depth) - 1u;
        if (cut_off > max_code) {
            throw std::invalid_argument(
                "AutoWhiteBalance cut_off exceeds the sensor code range");
        }
        return cut_off == 0 ? static_cast<uint16_t>(max_code) : cut_off;
    }

    AwbAlgorithm algo_;
    WhiteBalanceGains gains_; 
    uint16_t cut_off_;

    // Persistent device buffers, allocated lazily and reused across frames.
    unsigned long long* d_sums_  = nullptr;  // [4] R, Gr, Gb, B sums (GrayWorld)
    float*              d_gains_ = nullptr;  // [4] r, gr, gb, b gains
    bool                manual_uploaded_ = false;
};

std::unique_ptr<ISPBlock> createManualWhiteBalance(WhiteBalanceGains gains, int bit_depth,
                                                   uint16_t cut_off) {
    return std::make_unique<AutoWhiteBalance>(
        AwbAlgorithm::Manual, gains, bit_depth, cut_off);
}

std::unique_ptr<ISPBlock> createAutoWhiteBalance(int bit_depth, uint16_t cut_off) {
    // Gains are computed per-frame from image statistics, so the seed value is ignored.
    return std::make_unique<AutoWhiteBalance>(
        AwbAlgorithm::GrayWorld, WhiteBalanceGains{}, bit_depth, cut_off);
}
