#pragma once

#include "frame_buffer.h"
#include "sensor_config.h"
#include <cstdint>
#include <memory>
#include <string>

// ============================================================================
// FrameLoader — handles reading raw files from disk and transferring to GPU.
//
// Supports loading raw Bayer data (flat binary, uint16_t per pixel).
// Also supports downloading processed results back to host.
// ============================================================================
class FrameLoader {
public:
    FrameLoader();
    ~FrameLoader();

    // Load a raw Bayer file into a FrameBuffer on the GPU according to the
    // supplied SensorConfig. The returned FrameBuffer carries the original
    // packing (e.g. PACKED_10_MIPI); a downstream RawUnpack ISPBlock is
    // responsible for producing UNPACKED_U16 data.
    FrameBuffer load(const std::string& path, const SensorConfig& cfg);

    // Legacy entry point: load a flat unpacked uint16 raw file. Kept for
    // backward compatibility; new code should use load() with a SensorConfig.
    FrameBuffer loadRaw(const std::string& path, int width, int height,
                        PixelFormat format);

    // Download a GPU FrameBuffer to host memory.
    // Caller owns the returned buffer (sized to gpu_buf.sizeBytes()).
    // The pointer is typed as bytes; reinterpret_cast to the desired
    // element type (uint8_t / float / uint16_t) based on gpu_buf.format.
    std::unique_ptr<uint8_t[]> downloadToHost(const FrameBuffer& gpu_buf);

    // Save a RGB_U8 FrameBuffer as a PNG file.
    void savePNG(const FrameBuffer& gpu_buf, const std::string& path);

private:
    cudaStream_t stream_;
};
