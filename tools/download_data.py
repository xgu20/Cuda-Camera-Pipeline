#!/usr/bin/env python3
"""Download optional public datasets used by LibreCudaISP."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from convert_yml_to_json import convert_config


ROOT = Path(__file__).resolve().parents[1]
INFINITE_REPOSITORY = (
    "https://github.com/10x-Engineers/Infinite-ISP_ReferenceModel.git"
)
KAGGLE_DATASET = "kamilbryn/burst-raw-photography-dataset"
RAW_NAME_PATTERN = re.compile(
    r"(?P<width>\d+)x(?P<height>\d+)_(?P<bit_depth>\d+)bits?_"
    r"(?P<bayer>RGGB|BGGR|GRBG|GBRG)",
    re.IGNORECASE,
)


def require_command(command: str, install_hint: str) -> str:
    path = shutil.which(command)
    if path is None:
        raise SystemExit(f"Required command '{command}' was not found. {install_hint}")
    return path


def copy_file(source: Path, destination: Path, force: bool) -> bool:
    if destination.exists() and not force:
        print(f"Skipping existing file: {destination}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Copied {destination}")
    return True


def create_neutral_sidecar(raw_path: Path, force: bool) -> bool:
    """Create a runnable sidecar from an Infinite-ISP sample filename."""
    output = raw_path.with_suffix(".json")
    if output.exists() and not force:
        return False
    match = RAW_NAME_PATTERN.search(raw_path.stem)
    if match is None:
        print(f"Cannot infer metadata; no sidecar generated for {raw_path.name}")
        return False
    bit_depth = int(match.group("bit_depth"))
    sidecar = {
        "width": int(match.group("width")),
        "height": int(match.group("height")),
        "bit_depth": bit_depth,
        "bayer_pattern": match.group("bayer").upper(),
        "packing": "unpacked_u16" if bit_depth > 8 else "unpacked_u8",
        "black_level": 0,
        "white_level": (1 << bit_depth) - 1,
    }
    output.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"Generated neutral sidecar: {output}")
    return True


def download_infinite(data_root: Path, force: bool) -> None:
    git = require_command("git", "Install Git and retry.")
    destination = data_root / "infinite"
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="libreisp_infinite_") as temp:
        checkout = Path(temp) / "Infinite-ISP_ReferenceModel"
        subprocess.run(
            [git, "clone", "--depth", "1", INFINITE_REPOSITORY, str(checkout)],
            check=True,
        )
        sample_directory = checkout / "in_frames" / "normal" / "data"
        raw_files = sorted(sample_directory.glob("*.raw"))
        tuning_files = sorted(sample_directory.glob("*-configs.yml"))
        if not raw_files:
            raise SystemExit("The Infinite-ISP checkout did not contain sample RAW files.")
        for source in raw_files:
            copy_file(source, destination / source.name, force)
        for source in tuning_files:
            copy_file(source, destination / source.name, force)

    generated = []
    for yml_path in sorted(destination.glob("*-configs.yml")):
        output = yml_path.with_name(
            yml_path.name.removesuffix("-configs.yml") + ".json"
        )
        if output.exists() and not force:
            print(f"Skipping existing sidecar: {output}")
            continue
        generated.append(convert_config(yml_path, output))
        print(f"Converted {yml_path} -> {output}")
    neutral_count = sum(
        create_neutral_sidecar(raw_path, force)
        for raw_path in sorted(destination.glob("*.raw"))
    )
    print(
        f"Infinite-ISP: {len(raw_files)} RAW files, {len(tuning_files)} YAML files, "
        f"{len(generated)} tuned sidecars, and {neutral_count} neutral sidecars "
        f"are available in {destination}"
    )


def download_kaggle(data_root: Path, force: bool) -> None:
    kaggle = require_command(
        "kaggle",
        "Install and configure it with: python3 -m pip install kaggle",
    )
    destination = data_root / "kaggle"
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        kaggle,
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(destination),
        "--unzip",
    ]
    if force:
        command.append("--force")
    subprocess.run(command, check=True)
    print(f"Kaggle dataset is available in {destination}")
    print("Kaggle files may require format conversion and LibreCudaISP JSON sidecars.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        choices=("infinite", "kaggle", "all"),
        help="dataset source to download",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="destination root (default: repository data directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing files and force a fresh Kaggle download",
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()

    try:
        if args.source in ("infinite", "all"):
            download_infinite(data_root, args.force)
        if args.source in ("kaggle", "all"):
            download_kaggle(data_root, args.force)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Download command failed with exit code {exc.returncode}") from exc


if __name__ == "__main__":
    main()
