#include "blocks.h"
#include "isp_block.h"
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

// ============================================================================
// Sensor Linearization (OECF)
//
// Uses a per-channel lookup table (LUT) to linearize the sensor data.
// This corrects non-linearities in the sensor response.
// ============================================================================

// --- CUDA Kernel ---
__global__ void oecf_kernel(uint16_t* data, int width, int height,
                            const uint16_t* lut_r, const uint16_t* lut_gr,
                            const uint16_t* lut_gb, const uint16_t* lut_b,
                            PixelFormat format, int max_val) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint16_t val = data[idx];
    if (val > max_val) val = max_val;

    bool row_odd = (y & 1);
    bool col_odd = (x & 1);

    const uint16_t* channel_lut = nullptr;
    if (format == PixelFormat::BAYER_RGGB) {
        if (!row_odd && !col_odd) channel_lut = lut_r;
        else if (!row_odd && col_odd) channel_lut = lut_gr;
        else if (row_odd && !col_odd) channel_lut = lut_gb;
        else channel_lut = lut_b;
    } else if (format == PixelFormat::BAYER_BGGR) {
        if (!row_odd && !col_odd) channel_lut = lut_b;
        else if (!row_odd && col_odd) channel_lut = lut_gb;
        else if (row_odd && !col_odd) channel_lut = lut_gr;
        else channel_lut = lut_r;
    } else if (format == PixelFormat::BAYER_GRBG) {
        if (!row_odd && !col_odd) channel_lut = lut_gr;
        else if (!row_odd && col_odd) channel_lut = lut_r;
        else if (row_odd && !col_odd) channel_lut = lut_b;
        else channel_lut = lut_gb;
    } else { // BAYER_GBRG
        if (!row_odd && !col_odd) channel_lut = lut_gb;
        else if (!row_odd && col_odd) channel_lut = lut_b;
        else if (row_odd && !col_odd) channel_lut = lut_r;
        else channel_lut = lut_gr;
    }

    data[idx] = channel_lut[val];
}

// --- ISPBlock Implementation ---
class SensorLinearization : public ISPBlock {
public:
    explicit SensorLinearization(const std::vector<std::vector<uint16_t>>& lut, PixelFormat format, int out_bit_depth)
        : format_(format), out_bit_depth_(out_bit_depth), lut_size_(0),
          d_lut_r_(nullptr), d_lut_gr_(nullptr), d_lut_gb_(nullptr), d_lut_b_(nullptr) {
        
        if (lut.size() == 4) {
            lut_size_ = lut[0].size();
            for (int i = 1; i < 4; ++i) {
                if (lut[i].size() != lut_size_) {
                    throw std::invalid_argument("SensorLinearization all LUT channels must be the same size");
                }
            }

            size_t bytes = lut_size_ * sizeof(uint16_t);
            cudaMalloc(&d_lut_r_, bytes);
            cudaMalloc(&d_lut_gr_, bytes);
            cudaMalloc(&d_lut_gb_, bytes);
            cudaMalloc(&d_lut_b_, bytes);

            cudaMemcpy(d_lut_r_, lut[0].data(), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_lut_gr_, lut[1].data(), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_lut_gb_, lut[2].data(), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_lut_b_, lut[3].data(), bytes, cudaMemcpyHostToDevice);
        }
    }

    ~SensorLinearization() {
        if (d_lut_r_) cudaFree(d_lut_r_);
        if (d_lut_gr_) cudaFree(d_lut_gr_);
        if (d_lut_gb_) cudaFree(d_lut_gb_);
        if (d_lut_b_) cudaFree(d_lut_b_);
    }

    const char* name() const override { return "SensorLinearization"; }

    void process(const FrameBuffer& input, FrameBuffer& output,
                 cudaStream_t stream) override {
        if (!input.isBayer() || input.packing != PixelPacking::UNPACKED_U16) {
            throw std::invalid_argument(
                "SensorLinearization requires unpacked uint16 Bayer input");
        }

        // In-place operation — output aliases input
        output = input;
        
        // Update bit depth for downstream blocks
        output.bit_depth = out_bit_depth_;

        if (lut_size_ == 0) {
            return; // No LUT provided, pass-through
        }

        dim3 block(16, 16);
        dim3 grid((input.width + block.x - 1) / block.x,
                  (input.height + block.y - 1) / block.y);

        oecf_kernel<<<grid, block, 0, stream>>>(
            static_cast<uint16_t*>(output.d_data),
            input.width, input.height,
            d_lut_r_, d_lut_gr_, d_lut_gb_, d_lut_b_,
            format_, lut_size_ - 1);
    }

private:
    PixelFormat format_;
    int out_bit_depth_;
    size_t lut_size_;
    uint16_t *d_lut_r_, *d_lut_gr_, *d_lut_gb_, *d_lut_b_;
};

// Factory function
std::unique_ptr<ISPBlock> createSensorLinearization(const std::vector<std::vector<uint16_t>>& lut, PixelFormat format, int out_bit_depth) {
    return std::make_unique<SensorLinearization>(lut, format, out_bit_depth);
}
