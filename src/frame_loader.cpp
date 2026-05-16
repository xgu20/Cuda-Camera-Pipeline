#include "frame_loader.h"
#include <cstdio>
#include <fstream>
#include <stdexcept>

// stb_image_write for PNG output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

FrameLoader::FrameLoader() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

FrameLoader::~FrameLoader() {
    cudaStreamDestroy(stream_);
}

FrameBuffer FrameLoader::loadRaw(const std::string& path, int width, int height,
                                  PixelFormat format) {
    // Validate that it's a Bayer format
    FrameBuffer buf;
    buf.width    = width;
    buf.height   = height;
    buf.channels = 1;
    buf.format   = format;

    size_t pixel_count = static_cast<size_t>(width) * height;
    size_t file_size   = pixel_count * sizeof(uint16_t);
    buf.stride = static_cast<size_t>(width) * sizeof(uint16_t);

    // Read file into host memory
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open raw file: " + path);
    }

    // Allocate pinned host memory for fast DMA transfer
    uint16_t* h_data = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_data, file_size));

    file.read(reinterpret_cast<char*>(h_data), file_size);
    if (!file) {
        cudaFreeHost(h_data);
        throw std::runtime_error("Failed to read " + std::to_string(file_size) +
                                 " bytes from: " + path);
    }
    file.close();

    printf("[FrameLoader] Loaded %s (%dx%d, %zu bytes)\n",
           path.c_str(), width, height, file_size);

    // Allocate device memory and async copy
    CUDA_CHECK(cudaMalloc(&buf.d_data, file_size));
    CUDA_CHECK(cudaMemcpyAsync(buf.d_data, h_data, file_size,
                                cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Free pinned host buffer
    CUDA_CHECK(cudaFreeHost(h_data));

    return buf;
}

std::unique_ptr<uint8_t[]> FrameLoader::downloadToHost(const FrameBuffer& gpu_buf) {
    size_t total = gpu_buf.sizeBytes();
    auto host_buf = std::make_unique<uint8_t[]>(total);

    CUDA_CHECK(cudaMemcpyAsync(host_buf.get(), gpu_buf.d_data, total,
                                cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    return host_buf;
}

void FrameLoader::savePNG(const FrameBuffer& gpu_buf, const std::string& path) {
    if (gpu_buf.format != PixelFormat::RGB_U8) {
        throw std::runtime_error("savePNG requires RGB_U8 format");
    }

    auto host_buf = downloadToHost(gpu_buf);

    int stride_bytes = gpu_buf.width * 3;  // RGB, 1 byte each
    int ok = stbi_write_png(path.c_str(), gpu_buf.width, gpu_buf.height,
                            3, host_buf.get(), stride_bytes);

    if (!ok) {
        throw std::runtime_error("Failed to write PNG: " + path);
    }
    printf("[FrameLoader] Saved %s (%dx%d)\n",
           path.c_str(), gpu_buf.width, gpu_buf.height);
}
