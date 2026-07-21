#!/usr/bin/env python3
"""Run a directory of RAW files through the normal ISP pipeline.

Denoise and edge enhancement are enabled by default using sensor/golden tuning.
An explicit comparison mode is available without modifying source sidecars.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent


def set_enabled(config: dict, module: str, legacy_key: str, enabled: bool) -> None:
    """Set enable in the schema actually used by this sidecar.

    A module-local ``enabled`` has precedence over the legacy flat flag, so it
    must be updated when present. Otherwise retain the legacy representation
    for compatibility with the existing data set.
    """
    modules = config.get("modules")
    if isinstance(modules, dict) and isinstance(modules.get(module), dict):
        modules[module]["enabled"] = enabled
    elif isinstance(config.get(module), dict):
        config[module]["enabled"] = enabled
    else:
        config[legacy_key] = enabled


def run_variant(raw_file: Path, source_config: dict, output: Path,
                denoise: bool, edge: bool) -> None:
    config = dict(source_config)
    # Deep-copy nested module dictionaries before changing enable flags.
    config = json.loads(json.dumps(config))
    set_enabled(config, "yuv_denoise", "enable_yuv_denoise", denoise)
    set_enabled(config, "edge_enhancement", "enable_edge_enhancement", edge)

    with tempfile.TemporaryDirectory(prefix="libreisp_run_all_") as temp:
        temp_dir = Path(temp)
        temp_raw = temp_dir / raw_file.name
        temp_json = temp_raw.with_suffix(".json")
        temp_raw.symlink_to(raw_file.resolve())
        temp_json.write_text(json.dumps(config, indent=2) + "\n")

        env = os.environ.copy()
        env["GOLDEN_TUNING_FILE"] = str(ROOT / "config/golden_tuning.json")
        subprocess.run(
            [str(ROOT / "build/libreisp"), str(temp_raw), str(output),
             "--config", str(temp_json)],
            cwd=ROOT,
            env=env,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data/infinite",
        help="directory containing RAW files (default: data/infinite)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="one shared JSON config for every matched RAW file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/run_all",
        help="output directory (default: artifacts/run_all)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also render a baseline with denoise and edge enhancement disabled",
    )
    parser.add_argument(
        "--disable-edge",
        action="store_true",
        help="keep denoise enabled but disable edge enhancement",
    )
    parser.add_argument(
        "--pattern",
        default="*.raw",
        help="RAW filename glob under --input-dir (default: *.raw)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="search subdirectories and preserve their layout in the output",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = args.input_dir.resolve()
    shared_config = args.config.resolve() if args.config else None
    if shared_config and not shared_config.is_file():
        raise FileNotFoundError(f"Shared config does not exist: {shared_config}")
    raw_files = sorted(
        input_dir.rglob(args.pattern) if args.recursive
        else input_dir.glob(args.pattern)
    )
    if not raw_files:
        raise SystemExit(f"No RAW files matched {args.pattern!r}")

    for index, raw_file in enumerate(raw_files, 1):
        json_file = shared_config or raw_file.with_suffix(".json")
        if not json_file.exists():
            raise FileNotFoundError(f"Missing sidecar for {raw_file}: {json_file}")
        source_config = json.loads(json_file.read_text())
        relative_stem = raw_file.relative_to(input_dir).with_suffix("")
        output = output_dir / relative_stem.with_suffix(".png")
        output.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{index}/{len(raw_files)}] running: {relative_stem}", flush=True)
        run_variant(
            raw_file, source_config, output,
            denoise=True, edge=not args.disable_edge,
        )

        if args.compare:
            print(f"[{index}/{len(raw_files)}] baseline: {relative_stem}", flush=True)
            run_variant(
                raw_file, source_config,
                output.with_name(f"{output.stem}_baseline.png"),
                denoise=False, edge=False,
            )

    print(f"All done. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
