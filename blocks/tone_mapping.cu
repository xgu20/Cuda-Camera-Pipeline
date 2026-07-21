#include "blocks.h"
#include "isp_block.h"

#include <memory>
#include <stdexcept>

// ACES Filmic curve fit by Stephen Hill
__device__ __forceinline__ float3 aces_filmic(float3 x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    
    float3 num = make_float3(
        x.x * (a * x.x + b),
        x.y * (a * x.y + b),
        x.z * (a * x.z + b)
    );
    float3 den = make_float3(
        x.x * (c * x.x + d) + e,
        x.y * (c * x.y + d) + e,
        x.z * (c * x.z + d) + e
    );
    
    return make_float3(
        fminf(fmaxf(num.x / den.x, 0.0f), 1.0f),
        fminf(fmaxf(num.y / den.y, 0.0f), 1.0f),
        fminf(fmaxf(num.z / den.z, 0.0f), 1.0f)
    );
}

__global__ void toneMappingKernel(float* rgb, int pixel_count, float exposure) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;

    float* p = rgb + pixel * 3;
    float3 color = make_float3(p[0], p[1], p[2]);
    
    // Apply exposure compensation
    color.x *= exposure;
    color.y *= exposure;
    color.z *= exposure;
    
    // Apply ACES Filmic Tone Mapping
    color = aces_filmic(color);
    
    p[0] = color.x;
    p[1] = color.y;
    p[2] = color.z;
}

class ToneMappingBlock : public ISPBlock {
public:
    explicit ToneMappingBlock(float exposure) : exposure_(exposure) {}

    const char* name() const override { return "GlobalToneMapping (ACES)"; }

    void process(const FrameBuffer& input, FrameBuffer& output,
                 cudaStream_t stream) override {
        if (input.format != PixelFormat::RGB_FLOAT || input.channels != 3) {
            throw std::invalid_argument(
                "ToneMapping requires 3-channel RGB_FLOAT input");
        }

        output = input; // Zero-copy, in-place modification
        const int pixel_count = input.width * input.height;
        constexpr int threads = 256;
        const int blocks = (pixel_count + threads - 1) / threads;
        
        toneMappingKernel<<<blocks, threads, 0, stream>>>(
            static_cast<float*>(output.d_data), pixel_count, exposure_);
    }

private:
    float exposure_;
};

std::unique_ptr<ISPBlock> createToneMapping(float exposure) {
    return std::make_unique<ToneMappingBlock>(exposure);
}
