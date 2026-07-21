# Data and calibration files

The repository intentionally does not ship RAW images, downloaded archives,
or private sensor calibration. The root `.gitignore` ignores everything under
`data/` except this file.

Expected local layout:

```text
data/
├── README.md
├── synthetic/       generated inputs
├── kaggle/          public downloaded datasets
└── infinite/        public Infinite-ISP samples and calibration
```

Every RAW input requires a JSON sidecar with the same stem:

```text
scene.raw
scene.json
```

Alternatively, pass one shared JSON explicitly. This is useful for video or
burst frames captured with identical dimensions, packing, Bayer layout, black
level, and calibration:

```bash
./build/libreisp scene.raw scene.png --config configs.json
```

The explicit config overrides same-stem sidecar discovery. Do not share it
between captures whose sensor format or calibration differs.

## Option 1: generate synthetic data

This is the quickest way to run the project without downloading a dataset:

```bash
python3 -m pip install -r requirements-data.txt
mkdir -p data/synthetic
python3 tools/synthetic_gen.py \
  --input /path/to/input.png \
  --output data/synthetic/example.raw \
  --width 1920 \
  --height 1080
```

The generator writes both `example.raw` and `example.json`.

## Download helper

The repository provides one entry point for both public sources:

```bash
python3 -m pip install -r requirements-data.txt
python3 tools/download_data.py infinite
python3 tools/download_data.py kaggle
python3 tools/download_data.py all
```

Existing files are preserved by default. Pass `--force` to replace them, or
`--data-root /another/path` to use another destination. The Kaggle CLI must be
authenticated before downloading from Kaggle.

## Option 2: Burst RAW Photography dataset

The public
[Burst RAW Photography dataset on Kaggle](https://www.kaggle.com/datasets/kamilbryn/burst-raw-photography-dataset)
can be stored under `data/kaggle/`. Review and accept the dataset's current
terms on Kaggle before downloading it.

The helper runs the equivalent of the following with a configured Kaggle CLI:

```bash
mkdir -p data/kaggle
kaggle datasets download \
  -d kamilbryn/burst-raw-photography-dataset \
  -p data/kaggle
unzip data/kaggle/burst-raw-photography-dataset.zip -d data/kaggle
```

Files from external datasets are not necessarily in this pipeline's sidecar
format. Create a matching JSON sidecar and, when necessary, convert the source
packing before running `libreisp`.

## Option 3: Infinite-ISP sample captures

The `Indoor1` and `Outdoor1` through `Outdoor4` captures come from the public
[10xEngineers Infinite-ISP ReferenceModel data directory](https://github.com/10x-Engineers/Infinite-ISP_ReferenceModel/tree/main/in_frames/normal/data).
Each 10-bit GRBG RAW file has a matching tuned YAML configuration in that
directory.

The helper clones upstream into a temporary directory, copies the sample RAW
and YAML files, and generates compatible JSON sidecars:

```bash
python3 tools/download_data.py infinite
```

Rerun conversion after editing an upstream YAML file with:

```bash
python3 tools/convert_yml_to_json.py data/infinite
```

The upstream project is licensed under
[Apache-2.0](https://github.com/10x-Engineers/Infinite-ISP_ReferenceModel/blob/main/LICENSE).
Keep the upstream attribution and review those files before redistributing the
captures or derived calibration profiles. This project downloads nothing
automatically and does not vendor the samples.

Expected files may include:

```text
data/infinite/Indoor1_2592x1536_10bit_GRBG.raw
data/infinite/Indoor1_2592x1536_10bit_GRBG-configs.yml
data/infinite/Indoor1_2592x1536_10bit_GRBG.json
```

## Sidecar format

Example:

```json
{
  "width": 2592,
  "height": 1536,
  "bit_depth": 10,
  "bayer_pattern": "GRBG",
  "packing": "unpacked_u16",
  "black_level": 50,
  "white_level": 1023,
  "hot_pixel_threshold": 20,
  "dead_pixel_threshold": 20,
  "modules": {
    "white_balance": {
      "enabled": true,
      "mode": "gray_world"
    },
    "color_correction": {
      "enabled": true
    }
  },
  "color_correction_matrix": [
    2.203125, -1.08984375, -0.11328125,
    -0.29296875, 1.30859375, -0.015625,
    0.05859375, -0.88671875, 1.828125
  ]
}
```

The values above illustrate the schema and are not generic calibration values.
Use measurements appropriate for the actual sensor and lens.

## Tuning policy

`config/golden_tuning.json` is a versioned project default, not an Infinite-ISP
sensor calibration. It keeps sensor-dependent stages neutral or disabled and
provides generic defaults for the remaining pipeline.

Upstream YAML files mix sensor calibration, scene tuning, enable flags, and
settings for algorithms that do not map directly to this CUDA implementation.
`tools/convert_yml_to_json.py` converts the compatible subset: dimensions,
Bayer layout, packing, per-channel black levels, saturation, DPC threshold,
CCM, and enable flags. An upstream automatic-WB profile selects this project's
full-frame Gray World AWB; fixed gains are emitted only for upstream manual-WB
profiles. Statistics-window and exposure-filter settings are omitted because
the current CUDA AWB does not implement them. Unsupported algorithms are not
silently treated as equivalent. Remaining parameters follow the normal
precedence rule:

```text
downloaded sensor/scene sidecar > config/golden_tuning.json
```

If a locally obtained dataset has `*-configs.yml` files, synchronize supported
fields into JSON sidecars with:

```bash
python3 tools/convert_yml_to_json.py data/infinite
```

A single YAML file, including a generically named `configs.yml`, can be
converted directly and reused for a whole capture sequence:

```bash
python3 tools/convert_yml_to_json.py data/kaggle/capture/configs.yml
python3 run_all.py --input-dir data/kaggle/capture \
  --config data/kaggle/capture/configs.json --recursive
```

The helper is available at
[tools/convert_yml_to_json.py](../tools/convert_yml_to_json.py).
