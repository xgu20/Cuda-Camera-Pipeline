#!/usr/bin/env python3
"""Convert Infinite-ISP YAML tuning files to LibreCudaISP JSON sidecars."""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required: python3 -m pip install -r requirements-data.txt"
    ) from exc


def _enabled(section: dict[str, Any], default: bool = True) -> bool:
    return bool(section.get("is_enable", default))


def convert_config(yml_path: Path, output_path: Optional[Path] = None) -> Path:
    """Convert one upstream configuration and return the JSON output path."""
    with yml_path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source) or {}

    sensor = config.get("sensor_info", {})
    blc = config.get("black_level_correction", {})
    dpc = config.get("dead_pixel_correction", {})
    awb = config.get("auto_white_balance", {})
    wb = config.get("white_balance", {})
    ccm = config.get("color_correction_matrix", {})
    auto_wb = bool(wb.get("is_auto", False))
    wb_enabled = _enabled(wb) and (not auto_wb or _enabled(awb))

    bit_depth = int(sensor.get("bit_depth", 16))
    black_levels = [
        int(blc.get("r_offset", 0)),
        int(blc.get("gr_offset", blc.get("r_offset", 0))),
        int(blc.get("gb_offset", blc.get("r_offset", 0))),
        int(blc.get("b_offset", blc.get("r_offset", 0))),
    ]
    saturation = [
        int(blc.get("r_sat", (1 << bit_depth) - 1)),
        int(blc.get("gr_sat", (1 << bit_depth) - 1)),
        int(blc.get("gb_sat", (1 << bit_depth) - 1)),
        int(blc.get("b_sat", (1 << bit_depth) - 1)),
    ]
    threshold = int(dpc.get("dp_threshold", min(8000, (2 << bit_depth) - 1)))

    result: dict[str, Any] = {
        "width": int(sensor["width"]),
        "height": int(sensor["height"]),
        "bit_depth": bit_depth,
        "bayer_pattern": str(sensor["bayer_pattern"]).upper(),
        "packing": "unpacked_u16" if bit_depth > 8 else "unpacked_u8",
        "black_levels": black_levels,
        "black_level": min(black_levels),
        "white_level": min(saturation),
        "hot_pixel_threshold": threshold,
        "dead_pixel_threshold": threshold,
        "modules": {
            "black_level": {"enabled": _enabled(blc)},
            "dead_pixel_correction": {"enabled": _enabled(dpc)},
            "white_balance": {
                "enabled": wb_enabled,
                "mode": "gray_world" if auto_wb else "manual",
            },
            "color_correction": {"enabled": _enabled(ccm)},
        },
    }

    # Auto mode intentionally omits fixed gains so LibreCudaISP runs its Gray World
    # AWB. Upstream exposure filtering and statistics windows do not have
    # equivalents in the current CUDA implementation.
    if wb_enabled and not auto_wb:
        result["white_balance_gains"] = {
            "r": float(wb.get("r_gain", 1.0)),
            "gr": 1.0,
            "gb": 1.0,
            "b": float(wb.get("b_gain", 1.0)),
        }

    if _enabled(ccm) and ccm:
        rows = [
            ccm.get("corrected_red", [1024, 0, 0]),
            ccm.get("corrected_green", [0, 1024, 0]),
            ccm.get("corrected_blue", [0, 0, 1024]),
        ]
        flat_ccm = [float(value) for row in rows for value in row]
        # Older Infinite-ISP profiles stored Q10 integers; current profiles
        # store floating-point coefficients directly.
        scale = 1024.0 if any(abs(value) > 16.0 for value in flat_ccm) else 1.0
        result["color_correction_matrix"] = [value / scale for value in flat_ccm]

    if output_path is None:
        suffix = "-configs.yml"
        name = yml_path.name
        output_name = name[: -len(suffix)] + ".json" if name.endswith(suffix) else yml_path.stem + ".json"
        output_path = yml_path.with_name(output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output_path


def convert_directory(directory: Path) -> list[Path]:
    outputs = []
    for yml_path in sorted(directory.glob("*-configs.yml")):
        output = convert_config(yml_path)
        print(f"Converted {yml_path} -> {output}")
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("data/infinite"),
        help="a YAML file or directory containing *-configs.yml files",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="output JSON path (only valid when input is a file)",
    )
    args = parser.parse_args()
    if args.input.is_file():
        output = convert_config(args.input, args.output)
        print(f"Converted {args.input} -> {output}")
        return
    if args.output is not None:
        parser.error("--output is only valid when input is a file")
    outputs = convert_directory(args.input)
    if not outputs:
        raise SystemExit(f"No *-configs.yml files found in {args.input}")


if __name__ == "__main__":
    main()
