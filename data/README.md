# CUDA ISP Dataset and Calibration Data

This directory contains the raw sensor image files (`.raw`) and their corresponding metadata/calibration sidecars (`.json`), structured to support tests, calibration, and pipeline evaluation.

---

## Directory Structure

```
data/
├── README.md               # This documentation
├── test*.raw / .json      # Synthetic test patterns generated for pipeline unit tests
├── kaggle/                 # Large datasets (e.g. burst raw dataset zip)
└── infinite/               # Real-world sensor captures with calibration data
```

---

## Data Categories

### 1. Unit Test / Synthetic Data (Root Folder)

These files are generated dynamically or synthetically to test specific packing layouts, resolutions, and raw bit depths in the GoogleTest suite.

| Filename | Bayer Format | Packing | Bit Depth | Purpose |
|---|---|---|---|---|
| `test.raw` | RGGB | `unpacked_u16` | 10-bit | Basic pipeline testing |
| `test_packed.raw` | RGGB | `mipi10` | 10-bit | MIPI CSI-2 RAW10 packing validation |
| `test_unpacked.raw` | RGGB | `unpacked_u16` | 10-bit | Reference output for unpacking tests |
| `test_ref.raw` | RGGB | `unpacked_u16` | 12-bit | Multi-bit depth test patterns |
| `test_rggb_1920x1080_10bit.raw` | RGGB | `unpacked_u16` | 10-bit | General pipeline benchmark target |

---

### 2. Real-World Scenes & Calibration (`infinite/` Folder)

This folder contains real-world captures from an image sensor along with their calibration profiles. 

#### Raw Scenes
Includes captured frames under various lighting conditions (Indoor vs. Outdoor) and bit-depth layouts (8-bit vs. 10-bit):
- **`Indoor1_2592x1536_10bit_GRBG.raw`**: Standard indoor test image.
- **`Indoor1_2592x1536_8bit_GRBG.raw`**: 8-bit variant.
- **`Outdoor1` - `Outdoor4`**: Various outdoor scenes at 8-bit and 10-bit.

#### Calibration Targets
- **`ColorChecker_2592x1536_10bit_GRBG.raw`**: Raw capture of a standard 24-patch ColorChecker chart, used for tuning White Balance and generating the Color Correction Matrix (CCM).

---

## Calibration Config Formats

> [!IMPORTANT]
> The ISP pipeline expects metadata and calibration parameters in a `.json` sidecar next to the raw file (e.g. `Indoor1.raw` -> `Indoor1.json`).

### 1. YAML Config (`*-configs.yml`)
The dataset originally contains YAML configs specifying raw sensor properties, DPC thresholds, BLC offsets, AWB parameters, white balance gains, CCM, and Gamma LUTs.
Example parameters from `Indoor1_2592x1536_10bit_GRBG-configs.yml`:
- Black level: `50`
- Manual WB Gains: `R: 1.449`, `B: 3.242`
- Color Correction Matrix (CCM) scaled by 1024.

### 2. Sidecar JSON (`*.json`)
Converted automatically from YAML config. This JSON sidecar is read by the ISP during runtime:
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
  "white_balance_gains": {
    "r": 1.44921875,
    "gr": 1.0,
    "gb": 1.0,
    "b": 3.2421875
  },
  "color_correction_matrix": [
    2.203125, -1.08984375, -0.11328125,
    -0.29296875, 1.30859375, -0.015625,
    0.05859375, -0.88671875, 1.828125
  ]
}
```

> [!NOTE]
> Values in `color_correction_matrix` are floating-point coefficients derived by dividing the integer matrix values in the YAML file by `1024.0`.

---

## How to Convert/Update Configurations

A python helper script is provided at [convert_yml_to_json.py](file:///home/gxh1991/cuda_isp/tools/convert_yml_to_json.py) to sync YAML config settings to JSON sidecars. Run this script from the workspace root:

```bash
python3 tools/convert_yml_to_json.py
```
This updates all `.json` files under `data/infinite/` that have corresponding `-configs.yml` files.

---

## 公开数据集下载指引 (Public Datasets Download Guide)

为了保持 Git 仓库轻量，所有的 `.raw` 图像文件和 `.zip` 压缩包已在 `.gitignore` 中被忽略，不会提交到仓库中。您可以根据以下指引下载所需的公开数据集并放入相应目录：

### 1. Kaggle Burst RAW Photography Dataset
- **下载路径**: `data/kaggle/`
- **数据集来源**: [Burst RAW Photography Dataset (Kaggle)](https://www.kaggle.com/datasets/kamilbryn/burst-raw-photography-dataset)
- **命令行下载方式**:
  如果您配置了 Kaggle API 凭证，可以直接运行以下命令进行下载并解压：
  ```bash
  mkdir -p data/kaggle
  kaggle datasets download -d kamilbryn/burst-raw-photography-dataset -p data/kaggle/
  unzip data/kaggle/burst-raw-photography-dataset.zip -d data/kaggle/
  ```

### 2. Infinite Dataset (实拍与标定数据)
- **下载路径**: `data/infinite/`
- **数据集说明**: 包含 `Indoor1`, `Outdoor1-4` 等真实场景的 8-bit / 10-bit raw 图像及标定使用的 24 色卡图像。
- *(注：若是私有或特定实验室的数据源，请在此处补充您的网盘下载链接或存储桶地址。)*
