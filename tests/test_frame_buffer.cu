#include <gtest/gtest.h>

#include "frame_buffer.h"

TEST(FrameBufferTest, RejectsLayoutChangesOnAnExistingAllocation) {
    FrameBuffer frame;
    frame.width = 16;
    frame.height = 16;
    frame.channels = 1;
    frame.format = PixelFormat::BAYER_RGGB;
    frame.packing = PixelPacking::UNPACKED_U16;
    frame.allocate();

    EXPECT_EQ(frame.allocation_bytes, 16u * 16u * sizeof(uint16_t));

    frame.width = 32;
    EXPECT_THROW(frame.allocate(), std::runtime_error);

    frame.free();
    frame.allocate();
    EXPECT_EQ(frame.stride, 32u * sizeof(uint16_t));
    EXPECT_EQ(frame.allocation_bytes, 32u * 16u * sizeof(uint16_t));

    frame.free();
}
