#!/usr/bin/env python3
"""
Synthetic Bayer Data Generator

Takes a standard image (PNG/JPEG) and mosaics it into a raw Bayer RGGB file
for testing the CUDA ISP pipeline.

Usage:
    python3 synthetic_gen.py <input_image> <output.raw> [--width W] [--height H]

Example:
    python3 synthetic_gen.py test.png data/test_rggb_1920x1080_10bit.raw --width 1920 --height 1080

The output file is a flat binary of uint16 values (10-bit range, [0..1023]).
"""

import argparse
import json
import numpy as np
from pathlib import Path

def generate_color_bars(width: int, height: int) -> np.ndarray:
    """Generate a standard SMPTE-style color bar image (RGB float [0,1])."""
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    # 8 color bars
    colors = [
        [0.75, 0.75, 0.75],  # White (75%)
        [0.75, 0.75, 0.00],  # Yellow
        [0.00, 0.75, 0.75],  # Cyan
        [0.00, 0.75, 0.00],  # Green
        [0.75, 0.00, 0.75],  # Magenta
        [0.75, 0.00, 0.00],  # Red
        [0.00, 0.00, 0.75],  # Blue
        [0.00, 0.00, 0.00],  # Black
    ]
    
    bar_width = width // len(colors)
    for i, color in enumerate(colors):
        x_start = i * bar_width
        x_end = (i + 1) * bar_width if i < len(colors) - 1 else width
        img[:, x_start:x_end] = color
    
    return img

def generate_gradient(width: int, height: int) -> np.ndarray:
    """Generate a smooth horizontal gradient for each RGB channel."""
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    third_h = height // 3
    
    # Red gradient (top third)
    for x in range(width):
        img[:third_h, x, 0] = x / (width - 1)
    
    # Green gradient (middle third)
    for x in range(width):
        img[third_h:2*third_h, x, 1] = x / (width - 1)
    
    # Blue gradient (bottom third)
    for x in range(width):
        img[2*third_h:, x, 2] = x / (width - 1)
    
    return img

def rgb_to_bayer_rggb(rgb: np.ndarray, bit_depth: int = 10) -> np.ndarray:
    """
    Convert RGB image to RGGB Bayer pattern.
    
    RGGB layout:
      R  G  R  G  ...
      G  B  G  B  ...
      R  G  R  G  ...
    """
    h, w, _ = rgb.shape
    max_val = (1 << bit_depth) - 1
    
    bayer = np.zeros((h, w), dtype=np.uint16)
    
    # Even rows: R G R G ...
    bayer[0::2, 0::2] = (rgb[0::2, 0::2, 0] * max_val).astype(np.uint16)  # R
    bayer[0::2, 1::2] = (rgb[0::2, 1::2, 1] * max_val).astype(np.uint16)  # G
    
    # Odd rows: G B G B ...
    bayer[1::2, 0::2] = (rgb[1::2, 0::2, 1] * max_val).astype(np.uint16)  # G
    bayer[1::2, 1::2] = (rgb[1::2, 1::2, 2] * max_val).astype(np.uint16)  # B
    
    return bayer

def add_black_level(bayer: np.ndarray, black_level: int = 64) -> np.ndarray:
    """Add a black level offset to simulate real sensor behavior."""
    return (bayer.astype(np.uint32) + black_level).clip(0, 1023).astype(np.uint16)


def pack_mipi10(bayer: np.ndarray) -> np.ndarray:
    """Pack a 10-bit Bayer image (uint16) into MIPI CSI-2 RAW10 format.

    4 pixels are packed into 5 bytes per group:
        byte 0 = P0[9:2]
        byte 1 = P1[9:2]
        byte 2 = P2[9:2]
        byte 3 = P3[9:2]
        byte 4 = (P3[1:0]<<6) | (P2[1:0]<<4) | (P1[1:0]<<2) | P0[1:0]

    Returns a uint8 array of shape (height, ((width + 3) // 4) * 5).
    Width must currently be a multiple of 4.
    """
    h, w = bayer.shape
    assert w % 4 == 0, "MIPI10 packing requires width % 4 == 0"
    pixels = bayer.astype(np.uint16).reshape(h, w // 4, 4)

    high = (pixels >> 2).astype(np.uint8)             # (h, w/4, 4) — top 8 bits
    low  = (pixels & 0x3).astype(np.uint8)            # (h, w/4, 4) — bottom 2 bits

    packed_low = (
        (low[..., 0]      ) |
        (low[..., 1] << 2 ) |
        (low[..., 2] << 4 ) |
        (low[..., 3] << 6 )
    ).astype(np.uint8)                                # (h, w/4)

    out = np.empty((h, w // 4, 5), dtype=np.uint8)
    out[..., 0:4] = high
    out[..., 4]   = packed_low
    return out.reshape(h, -1)


def write_sidecar(json_path: Path, *,
                  width: int, height: int, bit_depth: int,
                  bayer_pattern: str, packing: str, black_level: int) -> None:
    """Write a JSON sidecar describing the raw file layout."""
    payload = {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "bayer_pattern": bayer_pattern,
        "packing": packing,
        "black_level": black_level,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Bayer test data")
    parser.add_argument("--output", "-o", type=str, default="data/test_rggb_1920x1080_10bit.raw",
                        help="Output raw file path")
    parser.add_argument("--width", "-W", type=int, default=1920, help="Image width")
    parser.add_argument("--height", "-H", type=int, default=1080, help="Image height")
    parser.add_argument("--pattern", type=str, default="colorbars",
                        choices=["colorbars", "gradient"],
                        help="Test pattern type")
    parser.add_argument("--black-level", type=int, default=64,
                        help="Black level offset to add (simulates sensor)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Optional input image (PNG/JPEG) to mosaic instead of test pattern")
    parser.add_argument("--packing", type=str, default="unpacked_u16",
                        choices=["unpacked_u16", "mipi10"],
                        help="Output bit-packing (unpacked uint16 or MIPI RAW10)")
    args = parser.parse_args()

    width = args.width
    height = args.height
    
    # Ensure even dimensions for Bayer pattern
    width = width - (width % 2)
    height = height - (height % 2)

    if args.input:
        try:
            from PIL import Image
            img = np.array(Image.open(args.input).resize((width, height))).astype(np.float32) / 255.0
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            print(f"Loaded input image: {args.input} -> {width}x{height}")
        except ImportError:
            print("PIL not available, using test pattern instead")
            img = generate_color_bars(width, height)
    elif args.pattern == "colorbars":
        img = generate_color_bars(width, height)
        print(f"Generated color bars: {width}x{height}")
    else:
        img = generate_gradient(width, height)
        print(f"Generated gradient: {width}x{height}")

    # Convert to Bayer
    bayer = rgb_to_bayer_rggb(img, bit_depth=10)
    print(f"  Bayer range before BLC: [{bayer.min()}, {bayer.max()}]")

    # Add black level
    bayer = add_black_level(bayer, args.black_level)
    print(f"  Bayer range after BLC={args.black_level}: [{bayer.min()}, {bayer.max()}]")

    # Pack (if requested) and save raw bytes
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.packing == "mipi10":
        if width % 4 != 0:
            raise SystemExit("mipi10 packing requires --width divisible by 4")
        packed = pack_mipi10(bayer)
        packed.tofile(str(output_path))
        physical_format = "MIPI RAW10 (4 pixels per 5 bytes)"
    else:
        bayer.tofile(str(output_path))
        physical_format = "uint16 (unpacked, 10-bit value in low bits)"

    file_size = output_path.stat().st_size

    # Always emit a JSON sidecar so cuda_isp can pick it up automatically.
    sidecar_path = output_path.with_suffix(".json")
    write_sidecar(
        sidecar_path,
        width=width, height=height, bit_depth=10,
        bayer_pattern="RGGB", packing=args.packing,
        black_level=args.black_level,
    )

    print(f"  Saved raw:     {output_path} ({file_size} bytes)")
    print(f"  Saved sidecar: {sidecar_path}")
    print(f"  Dimensions: {width}x{height}")
    print(f"  Format: RGGB, 10-bit, {physical_format}")
    print(f"\nRun ISP pipeline with:")
    print(f"  ./build/cuda_isp {output_path} output.png")

if __name__ == "__main__":
    main()
