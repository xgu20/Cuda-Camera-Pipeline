#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <stdexcept>

// ============================================================================
// PixelFormat — logical color layout of a FrameBuffer
// ============================================================================
enum class PixelFormat {
    // Bayer patterns (single-channel, uint16_t when unpacked)
    BAYER_RGGB,
    BAYER_BGGR,
    BAYER_GRBG,
    BAYER_GBRG,

    // Processed formats (3-channel, float)
    RGB_FLOAT,   // [0.0, 1.0] per channel
    YUV_FLOAT,

    // Output formats (3-channel, uint8_t)
    RGB_U8,
};

// ============================================================================
// PixelPacking — physical bit-packing of a Bayer FrameBuffer.
// Only meaningful when format is one of the BAYER_* values; for RGB / YUV
// formats this field is ignored and should stay UNPACKED_U16.
//
// MIPI CSI-2 RAW10 layout (4 pixels per 5 bytes):
//   byte 0 = P0[9:2]
//   byte 1 = P1[9:2]
//   byte 2 = P2[9:2]
//   byte 3 = P3[9:2]
//   byte 4 = (P3[1:0]<<6) | (P2[1:0]<<4) | (P1[1:0]<<2) | P0[1:0]
// ============================================================================
enum class PixelPacking {
    UNPACKED_U8,     // 8 bits per pixel; requires bit_depth <= 8
    UNPACKED_U16,    // 16 bits per pixel; low `bit_depth` bits valid
    PACKED_10_MIPI,  // 4 pixels packed into 5 bytes (MIPI CSI-2 RAW10)
};

// ============================================================================
// FrameBuffer — unified data container passed between ISP blocks
// ============================================================================
struct FrameBuffer {
    void*        d_data    = nullptr;  // Device (GPU) pointer
    int          width     = 0;
    int          height    = 0;
    int          channels  = 1;        // 1 for Bayer, 3 for RGB/YUV
    size_t       stride    = 0;        // Bytes per row
    size_t       allocation_bytes = 0; // Known allocation size; 0 if unknown/external
    PixelFormat  format    = PixelFormat::BAYER_RGGB;
    PixelPacking packing   = PixelPacking::UNPACKED_U16;
    int          bit_depth = 16;       // valid bits in each pixel value

    // ------- Helpers -------

    // Total bytes occupied by the buffer
    size_t sizeBytes() const {
        return stride * static_cast<size_t>(height);
    }

    // Logical bytes per pixel element when unpacked (used for non-Bayer formats
    // and for sizing unpacked Bayer buffers). For PACKED_10_MIPI the row stride
    // is computed separately by computeRowBytes() below.
    size_t elementSize() const {
        switch (format) {
            case PixelFormat::BAYER_RGGB:
            case PixelFormat::BAYER_BGGR:
            case PixelFormat::BAYER_GRBG:
            case PixelFormat::BAYER_GBRG:
                return sizeof(uint16_t);
            case PixelFormat::RGB_FLOAT:
            case PixelFormat::YUV_FLOAT:
                return sizeof(float);
            case PixelFormat::RGB_U8:
                return sizeof(uint8_t);
        }
        return 0;
    }

    // Is this a Bayer (pre-demosaic) format?
    bool isBayer() const {
        return format == PixelFormat::BAYER_RGGB ||
               format == PixelFormat::BAYER_BGGR ||
               format == PixelFormat::BAYER_GRBG ||
               format == PixelFormat::BAYER_GBRG;
    }

    // Computes the number of bytes per row, taking PixelPacking into account.
    // Bayer + PACKED_10_MIPI requires width to be a multiple of 4.
    size_t computeRowBytes() const {
        if (isBayer()) {
            switch (packing) {
                case PixelPacking::UNPACKED_U8:
                    return static_cast<size_t>(width) * sizeof(uint8_t);
                case PixelPacking::UNPACKED_U16:
                    return static_cast<size_t>(width) * sizeof(uint16_t);
                case PixelPacking::PACKED_10_MIPI:
                    return ((static_cast<size_t>(width) + 3) / 4) * 5;
            }
        }
        return static_cast<size_t>(width) * channels * elementSize();
    }

    // Allocate GPU memory matching this buffer's dimensions / format / packing.
    void allocate() {
        const size_t required_stride = computeRowBytes();
        const size_t total = required_stride * static_cast<size_t>(height);
        if (d_data) {
            if (allocation_bytes == total && stride == required_stride) {
                return;
            }
            throw std::runtime_error(
                "FrameBuffer::allocate layout mismatch; owning code must free "
                "the existing allocation before reusing this view");
        }
        stride = required_stride;
        cudaError_t err = cudaMalloc(&d_data, total);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                     cudaGetErrorString(err));
        }
        allocation_bytes = total;
    }

    // Free GPU memory. FrameBuffer is a non-owning view: there is no destructor,
    // so whoever called allocate() (or otherwise produced the d_data) is
    // responsible for calling free() exactly once when the buffer is no longer
    // needed. Copies are shallow and share d_data.
    void free() {
        if (d_data) {
            cudaFree(d_data);
            d_data = nullptr;
            allocation_bytes = 0;
        }
    }
};

// ============================================================================
// CUDA error checking macro
// ============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                    \
            throw std::runtime_error(cudaGetErrorString(err));                  \
        }                                                                       \
    } while (0)
