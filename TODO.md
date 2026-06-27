# cuda_isp — TODO

来自一次代码 review 的待办清单，按优先级（正确性 → 性能 → 工程化）排序。
完成后请把对应条目从 `- [ ]` 改成 `- [x]`，或者直接删除。

---

## 一、正确性 & 设计（优先级最高）

- [ ] **标定数据与多光源 CCM** — 当前已支持 sidecar 传入单个 3×3 CCM；后续
  增加 ColorChecker 标定工具，以及按色温在两组标定矩阵间插值。

- [ ] **析构清理错误可观测性** — `FrameBuffer::free()` / pipeline / AWB 析构
  当前析构路径不会抛异常，但会静默忽略 `cudaFree` / `cudaStreamDestroy`
  错误。增加专用 noexcept cleanup helper，失败时记录日志。

- [ ] **`FrameBuffer::stride` 语义** — 当前总是紧密排列的 row bytes，没有
  padding 含义。要么改名 `row_bytes`，要么真正用 `cudaMallocPitch` 支持
  pitched memory，并让所有 block 使用 stride。


---

## 二、ISP Blocks 路线图

### P0 — 真实传感器基础链路

- [ ] **Per-channel Black Level Correction** — 当前 BLC 只有单一 offset；扩展为
  `R / Gr / Gb / B` 四通道 black level，并支持 optical-black 区域统计与行级
  black-level 漂移校正。

- [ ] **Sensor Linearization / OECF** — 在 BLC 后用每 Bayer 通道 LUT 修正传感器
  非线性响应；sidecar 支持 LUT 与输入/输出 bit depth。

- [ ] **Lens Shading Correction (LSC)** — 在 Bayer 域应用四通道二维 gain map，
  修正镜头暗角与 color shading；支持标定网格的双线性插值。

- [ ] **Dead / Defective Pixel Correction (DPC)** — Demosaic 前检测 hot/dead pixel，
  使用同色 Bayer 邻居恢复；避免坏点被 demosaic 扩散成彩色斑点。

- [ ] **Digital Gain / Exposure Compensation** — 在 Bayer linear 域应用可配置 gain，
  明确饱和、定点精度及与 AWB gain 的执行顺序；为后续 AE 留出控制接口。

- [ ] **Bayer-domain Noise Reduction (BNR)** — Demosaic 前做 CFA-aware spatial
  denoise，结合 ISO/noise profile 调节强度，并保护边缘与细节。

### P1 — 颜色与成像质量

- [ ] **Edge-aware Demosaic** — 在 bilinear 基线之外实现 Malvar-He-Cutler 或
  directional/edge-aware 版本，降低 zipper、false color 与高频细节损失。

- [ ] **Robust AWB Statistics** — GrayWorld 增加统计 ROI、欠曝/过曝排除、
  gain clamp、置信度与跨帧平滑；避免大面积单色场景导致严重偏色。

- [ ] **Auto Exposure Statistics / Control** — 增加 histogram、亮度 ROI、曝光目标
  与跨帧控制接口；算法输出 exposure time / analog gain / digital gain 建议值。

- [ ] **Tone Mapping / Contrast Curve** — 在 linear RGB/亮度域实现曝光补偿、
  highlight roll-off、shadow lift 和可配置 tone curve，再进入显示 gamma。

- [ ] **RGB-domain Denoise** — Demosaic 后抑制彩噪与残余亮度噪声；与 BNR 分工，
  并针对高 ISO 配置强度。

- [ ] **Sharpen / Detail Enhancement** — 输出前增加 edge-aware unsharp mask，
  包含阈值、halo 抑制与噪声保护。

- [ ] **Chromatic Aberration Correction** — 校正横向色差，避免画面边缘 R/B
  通道错位；参数来自镜头标定。

### P2 — 输出与几何处理

- [ ] **Crop / Invalid Region Removal** — 支持 sensor active area 与用户 ROI，
  同时正确更新 Bayer phase，避免奇数 offset 导致 pattern 错位。

- [ ] **Resize / Scale** — 支持高质量 RGB resize；如需 Bayer-domain resize，
  必须保持 CFA pattern 与采样位置。

- [ ] **RGB → YUV / NV12 Output** — 支持 BT.601 / BT.709、full/limited range，
  以及常用 NV12/YUV420 输出，便于接视频编码器。

- [ ] **Output Quantization + Dithering** — 从 float/高 bit-depth 输出到 8-bit 时
  增加可选 dithering，减少平滑渐变 banding。

### P3 — 视频、HDR 与进阶功能

- [ ] **Temporal Noise Reduction (TNR)** — 跨帧降噪，包含运动检测/对齐、
  history 管理与 ghosting 抑制。

- [ ] **HDR Merge** — 合并多曝光 RAW，处理运动区域、饱和像素与曝光比例，
  输出高动态范围 linear Bayer/RGB。

- [ ] **Local Tone Mapping** — 在 HDR/高动态场景增强局部对比度，同时控制 halo
  与时域闪烁。

- [ ] **3D LUT / Look Processing** — CCM 与 tone mapping 后增加可选 3D LUT，
  用于显示变换或风格化 look。

---

## 三、CUDA 性能优化

- [ ] **融合 Gamma + OutputPack** — `blocks/gamma.cu` + `blocks/output_pack.cu`
  目前 `float → gamma(float) → u8` 是两个 kernel，中间多一轮 95 MB（4K）
  R/W。融合成 `float → gamma + pack → u8` 单 kernel，预计末段时间砍半。

- [ ] **融合 BLC + Demosaic** — `blocks/black_level.cu` + `blocks/demosaic.cu`
  Demosaic kernel 读取邻居像素时直接做 `max(0, val - black_level)`，省掉
  BLC 的一轮全量 R/W。

- [ ] **Demosaic optimized SMEM bank conflict** — `blocks/demosaic.cu`
  ```cpp
  __shared__ uint16_t smem[SMEM_H][SMEM_W];   // 2-byte stride → 2-way conflict
  ```
  方案：列方向 padding 到 `SMEM_W + 1`，或者改成 `uint32_t` 一槽位存两像素。

- [ ] **Demosaic 边缘 mirror padding** — `blocks/demosaic.cu`
  当前边缘 clamp 到自身导致 1 像素宽色边。改成 `x-1<0 ? x+1 : x-1` 镜像。

- [ ] **`ISPPipeline::execute` 末尾一次性 sync** — `src/isp_pipeline.cpp`
  ```cpp
  for (block in blocks) { record(start); launch; record(stop); sync; elapsed; }
  ```
  当前每个 block 都 sync，串行化 launch 开销。改成先记录所有 events，
  最后一次 sync + 批量 `cudaEventElapsedTime`。

- [ ] **BLC naive kernel 1D launch** — `blocks/black_level.cu`
  数据是连续 1D 的，没必要 2D `(blockIdx.y, threadIdx.y)`。改 1D 省寄存器
  + 边界检查（参考 optimized 版本的范式）。

---

## 四、测试 / 工程化

- [ ] **抽取测试 helper** — `tests/test_*.cu`
  三个文件里 `Performance_4K`（events / warm-up / loop / GB 计算）几乎逐
  行重复。抽成模板 helper 或 GoogleTest 参数化 fixture。

- [ ] **Performance 测试别用 `cudaMemset(0)`** — `tests/test_gamma.cu`
  全 0 输入下 `srgb_gamma` 永远走 `x <= 0.0031308` 快分支，测不出 `__powf`
  真实开销。改用 `curandGenerateUniform` 或一次性 host 填真实数据。

- [ ] **`execute()` 与 `printSummary()` 重复输出** — `src/isp_pipeline.cpp`
  两边都打印每 block 时间。要么 `execute` 静默 + `printSummary` 输出明细，
  要么 `execute` 输出明细 + `printSummary` 只输出 TOTAL。

- [ ] **`FrameLoader::loadRaw` 简化** — `src/frame_loader.cpp`
  单次同步加载下，pinned + async + sync 等价于普通 `cudaMemcpy`，没有
  overlap 收益。直接 `cudaMemcpy` 简单一些，将来要 streaming 再引入
  pinned buffer 池。

---

## 五、杂项

- [ ] **`synthetic_gen.py` 边界** — `tools/synthetic_gen.py`
  `width == 1` 时除以 0；`height < 3` 时 `third_h == 0` 整图全黑。改用
  `np.linspace` 替代手写循环。

---

## 已完成

- [x] **根 CMakeLists.txt glob 加 `CONFIGURE_DEPENDS`** — `CMakeLists.txt`
      添加了 `CONFIGURE_DEPENDS`，现在新增/删除 `.cu` 文件会自动触发重新 cmake。
- [x] **README** — `README.md`
      补齐 build / run / test 三段说明，含 JSON sidecar 字段、`synthetic_gen.py`
      用法、GoogleTest filter 例子，以及开发注意事项（glob `CONFIGURE_DEPENDS`、
      `stride` 实为 `row_bytes`、pipeline 所有权契约）。

- [x] **抽取 `include/blocks.h` 集中 factory 声明** —
      避免每个调用方重复 `extern`，签名漂移变编译错误而非链接错误。
- [x] **`FrameLoader::downloadToHost` 文档/实现一致 + 异常安全** —
      改返回 `std::unique_ptr<uint8_t[]>`，干掉 `malloc/::free` 与文档
      "allocated with new[]" 不一致问题；`savePNG` 异常路径不再泄漏。
- [x] **Demosaic 多 Bayer pattern 支持** — RGGB/BGGR/GRBG/GBRG 全部支持，
      通过模板参数 `<RED_BX, RED_BY>` 编译期特化，零运行时分支；naive
      和 optimized 共用 `demosaicPixel<>` 逻辑。
- [x] **AWB full-scale clamp** — 修复 `1 << bit_depth - 1` 优先级错误，10-bit
      白平衡现在正确截到 1023 而非 512，并补 Manual AWB 回归测试。
- [x] **`ISPPipeline::execute` 输入/返回所有权** — 默认 `execute(FrameBuffer&)`
      零拷贝并明确消费 input；需要保留输入时显式调用
      `executePreservingInput()` 使用可复用 staging buffer。返回值始终是非拥有
      视图，可能 alias input 或 pipeline-owned buffer。
- [x] **`BENCH_ITERS` 零拷贝** — benchmark 始终调用 `execute()`，不为保持
      unpacked_u16 输入内容一致而增加 staging copy；最终图片不作为 benchmark
      正确性结果。
- [x] **SensorConfig 关键边界校验** — 拒绝负 black level、错误 MIPI10 bit
      depth、非法 white level、奇数 Bayer 尺寸及非 finite WB gain。
- [x] **`FrameBuffer::allocate()` layout 变化安全** — 记录真实 allocation
      大小；浅拷贝 view 的尺寸或 format 与现有 allocation 不匹配时明确抛错，
      避免静默复用旧 buffer 越界写或误释放共享指针。
- [x] **Color Correction Matrix block** — Demosaic 后、Gamma 前应用可配置 3×3
      sensor RGB → target RGB 矩阵；未提供时使用 identity。
