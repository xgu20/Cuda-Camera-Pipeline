# cuda_isp — TODO

来自一次代码 review 的待办清单，按优先级（正确性 → 性能 → 工程化）排序。
完成后请把对应条目从 `- [ ]` 改成 `- [x]`，或者直接删除。

---

## 一、正确性 & 设计（优先级最高）

- [ ] **BLC optimized 算术安全** — `blocks/black_level_optimized.cu`
  `uint32_t - uint32_t` 隐式赋值给 `int`，以及 `p1 << 16` 越过符号位，都是
  实现定义/UB。改用显式 `int` cast，或者直接用 `__vsubss2` / `__vmaxs2`
  PTX SIMD intrinsic 一行搞定 16-bit 饱和减法 + clamp。

- [ ] **`ISPPipeline` 析构 noexcept** — `src/isp_pipeline.cpp`
  析构里调用的 `buf.free()` 内部走 `cudaFree`，但项目里 `CUDA_CHECK` 会
  throw。析构期间抛异常 = `std::terminate`。给析构路径专门写 noexcept
  版本（log + 吞掉错误）。

- [ ] **`FrameBuffer::allocate()` 行为修正** — `include/frame_buffer.h`
  - 当 `d_data` 已非空时静默 return，但 `width/height` 改了的话会保留旧
    buffer。要么 free + realloc，要么 `assert(d_data == nullptr)`。
  - `stride` 当前总是 `width * channels * elementSize`，没有 padding 含义。
    要么改名 `row_bytes`，要么真正用 `cudaMallocPitch` 支持 pitched 内存。


---

## 二、CUDA 性能优化

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

## 三、测试 / 工程化

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

## 四、杂项

- [ ] **`synthetic_gen.py` 边界** — `tools/synthetic_gen.py`
  `width == 1` 时除以 0；`height < 3` 时 `third_h == 0` 整图全黑。改用
  `np.linspace` 替代手写循环。

- [ ] **根 `CMakeLists.txt` glob 加 `CONFIGURE_DEPENDS`** — `CMakeLists.txt`
  当前根目录 glob 没加 `CONFIGURE_DEPENDS`（tests 那份加了），新增 `.cu`
  不会触发重新 cmake。顺便考虑把 blocks 抽成 OBJECT library，避免主程序
  和 isp_tests 各编译一遍。

- [ ] **README** — `README.md`
  目前只有标题。补 build / run / test 三段示例：
  ```bash
  cmake -B build && cmake --build build
  ./build/cuda_isp data/test_rggb_1920x1080_10bit.raw 1920 1080 output.png
  ctest --test-dir build
  ```

---

## 已完成

- [x] **抽取 `include/blocks.h` 集中 factory 声明** —
      避免每个调用方重复 `extern`，签名漂移变编译错误而非链接错误。
- [x] **`FrameLoader::downloadToHost` 文档/实现一致 + 异常安全** —
      改返回 `std::unique_ptr<uint8_t[]>`，干掉 `malloc/::free` 与文档
      "allocated with new[]" 不一致问题；`savePNG` 异常路径不再泄漏。
- [x] **Demosaic 多 Bayer pattern 支持** — RGGB/BGGR/GRBG/GBRG 全部支持，
      通过模板参数 `<RED_BX, RED_BY>` 编译期特化，零运行时分支；naive
      和 optimized 共用 `demosaicPixel<>` 逻辑。
- [x] **`ISPPipeline::execute` 返回值所有权** —
      `execute()` 末尾把最后一个 buffer 从 `intermediates_` pop 出来交给
      调用方，所有权契约写进 `isp_pipeline.h` 的 doc：返回的 d_data 与
      input 不同则调用方 `.free()`，否则是 input 的视图、不要 free。
