#!/usr/bin/env python3
"""Run every data/infinite RAW through the normal ISP pipeline.

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
            [str(ROOT / "build/libreisp"), str(temp_raw), str(output)],
            cwd=ROOT,
            env=env,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
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
        help="RAW filename glob under data/infinite (default: *.raw)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted((ROOT / "data/infinite").glob(args.pattern))
    if not raw_files:
        raise SystemExit(f"No RAW files matched {args.pattern!r}")

    for index, raw_file in enumerate(raw_files, 1):
        json_file = raw_file.with_suffix(".json")
        if not json_file.exists():
            raise FileNotFoundError(f"Missing sidecar for {raw_file}: {json_file}")
        source_config = json.loads(json_file.read_text())
        name = raw_file.stem

        print(f"[{index}/{len(raw_files)}] running: {name}", flush=True)
        run_variant(
            raw_file, source_config, output_dir / f"{name}.png",
            denoise=True, edge=not args.disable_edge,
        )

        if args.compare:
            print(f"[{index}/{len(raw_files)}] baseline: {name}", flush=True)
            run_variant(
                raw_file, source_config, output_dir / f"{name}_baseline.png",
                denoise=False, edge=False,
            )

    print(f"All done. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
