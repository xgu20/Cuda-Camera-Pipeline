# cuda_isp — CUDA Camera ISP Pipeline

一个用 CUDA 实现的相机 ISP（Image Signal Processor）流水线，把 raw Bayer 传感器数据处理成 RGB PNG。每个 ISP block 是独立的 `.cu` kernel，通过 `ISPPipeline` 串起来，自带逐 block 计时。

当前实现的 block：
- **raw_unpack** — MIPI10 / packed → uint16
- **black_level** — 黑电平校正（naive + optimized 两版）
- **auto_white_balance** — 白平衡：Manual（固定增益）+ GrayWorld（GPU 上统计每通道均值并计算增益，全程无 host 往返）
- **demosaic** — 双线性去马赛克，支持 RGGB / BGGR / GRBG / GBRG（编译期模板特化）
- **gamma** — sRGB gamma 校正（float）
- **output_pack** — float → uint8

## 目录结构

```
cuda_isp/
├── blocks/          # 每个 ISP block 一个 .cu 文件
├── include/         # 公共头文件（FrameBuffer / ISPBlock / pipeline 等）
├── src/             # 主程序 + pipeline driver + frame loader
├── tests/           # GoogleTest 单测 + Performance_4K 性能测试
├── tools/           # synthetic_gen.py（造测试 raw 数据）
├── data/            # 测试 raw + JSON sidecar
├── third_party/     # stb_image / stb_image_write
└── CMakeLists.txt
```

## 依赖

- CUDA Toolkit（>= 11，建议 12.x）
- CMake >= 3.25
- C++17 编译器
- Python 3 + numpy + Pillow（仅当用 `tools/synthetic_gen.py` 时）

`nlohmann/json` 和 `googletest` 由 CMake `FetchContent` 自动拉取，第一次 configure 时需要联网。

## Build

标准 out-of-source 构建：

```bash
cmake -B build
cmake --build build -j
```

说明：
- 默认走 **Release**（`CMakeLists.txt` 强制设置，避免空 build type 关掉 NDEBUG）。Debug 用 `cmake -B build -DCMAKE_BUILD_TYPE=Debug`。
- `CMAKE_CUDA_ARCHITECTURES native` 会自动探测当前机器 GPU 架构，无需手写 `-gencode`。
- 产物：`build/cuda_isp`（主程序）、`build/tests/isp_tests`（测试二进制）。
- `build/compile_commands.json` 已导出，根目录有软链接给 clangd 用。

## Run

```bash
./build/cuda_isp <input.raw> [output.png]
```

每个 `.raw` 必须配一个同名 `.json` sidecar（程序自动把 `.raw` 后缀替换为 `.json`），描述 sensor 元信息：

```json
{
  "width": 1920,
  "height": 1080,
  "bit_depth": 10,
  "bayer_pattern": "RGGB",
  "packing": "mipi10",
  "black_level": 64
}
```

字段说明：
- `bayer_pattern`: `RGGB` / `BGGR` / `GRBG` / `GBRG`
- `packing`: `mipi10`（5 字节 4 像素）/ `unpacked_u16`（每像素 2 字节小端 uint16）/ `unpacked_u8`（每像素 1 字节，要求 `bit_depth ≤ 8`）
- `bit_depth`: 1–16；常见 8 / 10 / 12
- `white_balance_gains`（可选）: `{ "r": .., "gr": .., "gb": .., "b": .. }`，默认全 `1.0`，须为正数；供 Manual 白平衡使用

例子：

```bash
./build/cuda_isp data/test_rggb_1920x1080_10bit.raw output.png
```

输出会打印每个 block 的耗时 + 总耗时。

设 `BENCH_ITERS=N` 可对同一帧连续跑 N 次，用于测稳态性能：第一帧支付一次性 buffer 分配，后续帧复用 pipeline 的 buffer 池（不再 `cudaMalloc`）。例如 `BENCH_ITERS=5 ./build/cuda_isp ...`。

### 造测试数据

用任意 PNG/JPEG mosaic 成 Bayer raw：

```bash
python3 tools/synthetic_gen.py input.png data/my_test.raw \
    --width 1920 --height 1080
```

脚本会顺带生成对应的 `.json` sidecar。

## Test

测试用 GoogleTest，通过 `gtest_discover_tests` 自动注册到 CTest，两种跑法：

```bash
# 用 ctest（推荐：并行 + summary）
ctest --test-dir build --output-on-failure
ctest --test-dir build -j8                    # 并行
ctest --test-dir build -R Demosaic            # 只跑名字含 Demosaic 的

# 直接跑 gtest 二进制（filter 更灵活，能看完整 stdout）
./build/tests/isp_tests
./build/tests/isp_tests --gtest_filter='*Performance_4K*'
./build/tests/isp_tests --gtest_list_tests
```

每个 block 都有 `Correctness_*` 单测 + `Performance_4K` 基准（带 warm-up + 多次 loop + GB/s 吞吐量），跑全套大概 10–30 秒。

## 开发笔记

- 想加新 block：在 `blocks/` 下放一个 `.cu`，在 `include/blocks.h` 里加 factory 声明，CMake 根目录的 `file(GLOB)` 会自动捡起来。**注意当前根目录 glob 没加 `CONFIGURE_DEPENDS`，新增 `.cu` 要手动重跑 `cmake -B build` 一次。**
- `FrameBuffer` 用 plain `cudaMalloc`，没有 pitched 内存；`stride` 字段当前其实是 `row_bytes`。
- Pipeline 的所有权契约：`ISPPipeline` 持有所有中间 buffer 并**跨帧复用**（buffer 池），只在输入几何尺寸变化时重新分配。`execute()` 返回的 `FrameBuffer` 是 pipeline 所拥有 buffer 的**非拥有视图**（若所有 block 都 in-place，则是 input 的视图），**调用方不得 `.free()`**；buffer 在下次 `execute()` 几何变化或 pipeline 析构时释放。

后续 TODO 见 `TODO.md`。
