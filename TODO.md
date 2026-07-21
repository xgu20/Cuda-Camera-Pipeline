# LibreCudaISP roadmap

Items are ordered by correctness, image quality, performance, and engineering
priority. Mark an item complete or remove it when the work lands.

## Correctness and design

- [ ] **Calibration tooling and multi-illuminant CCM** — Add a ColorChecker
  calibration tool and interpolate between matrices based on color temperature.
- [ ] **Observable cleanup failures** — Add noexcept cleanup helpers that log
  failures from `cudaFree` and `cudaStreamDestroy`.
- [ ] **Clarify `FrameBuffer::stride`** — Rename it to `row_bytes`, or implement
  pitched allocation and make every block honor the stride.

## ISP blocks

### P0: sensor-domain foundations

- [x] **Per-channel black-level correction** — Four Bayer-channel offsets are
  supported. Optical-black statistics and row drift remain future work.
- [x] **Sensor linearization / OECF** — Per-channel LUT correction after BLC.
- [x] **Lens-shading correction** — Four-channel 2D gain maps with interpolation.
- [x] **Dead-pixel correction** — Detect and replace hot/dead pixels before
  demosaic using same-color Bayer neighbors.
- [ ] **Digital gain / exposure compensation** — Define gain order, saturation,
  precision, and an interface for future AE control.
- [ ] **Bayer noise reduction** — Add CFA-aware, edge-preserving denoise driven
  by ISO/noise profiles.

### P1: color and image quality

- [ ] **Edge-aware demosaic** — Add Malvar-He-Cutler or a directional method to
  reduce zippering and false color.
- [ ] **Robust AWB statistics** — Add an ROI, exposure rejection, gain limits,
  confidence, and temporal smoothing.
- [ ] **Auto-exposure statistics and control** — Add histograms, metering ROIs,
  a target level, and exposure-time/analog-gain/digital-gain recommendations.
- [x] **Tone mapping** — Exposure, highlight roll-off, shadow lift, and a
  configurable curve before display gamma.
- [ ] **RGB-domain denoise** — Suppress residual luma and chroma noise after
  demosaic while complementing Bayer denoise.
- [ ] **Sharpening / detail enhancement** — Add edge-aware unsharp masking with
  thresholds, halo control, and noise protection.
- [ ] **Chromatic-aberration correction** — Correct lens-calibrated lateral R/B
  displacement near image edges.

### P2: output and geometry

- [ ] **Crop / active area** — Support sensor active regions and user ROIs while
  updating Bayer phase for odd offsets.
- [ ] **Resize** — Add high-quality RGB scaling; preserve CFA sample geometry if
  Bayer-domain scaling is introduced.
- [ ] **NV12 / YUV420 output** — Support BT.601/BT.709 and full/limited range.
- [ ] **Quantization and dithering** — Reduce banding when converting high-bit-
  depth or float data to 8-bit output.

### P3: video and HDR

- [ ] **Temporal noise reduction** — Add motion estimation, history management,
  and ghosting control.
- [ ] **HDR merge** — Merge bracketed RAW frames with motion and saturation
  handling.
- [ ] **Local tone mapping** — Improve local contrast while controlling halos
  and temporal flicker.
- [ ] **3D LUT / look processing** — Add an optional display or creative transform
  after CCM and tone mapping.

## CUDA performance

- [ ] **Fuse gamma and output packing** — Avoid the intermediate float buffer
  and its extra full-frame read/write pass.
- [ ] **Fuse BLC and demosaic** — Apply black subtraction while demosaic reads
  each Bayer neighbor.
- [ ] **Remove demosaic shared-memory bank conflicts** — Pad `SMEM_W` or pack two
  pixels into each `uint32_t` slot.
- [ ] **Mirror demosaic boundaries** — Replace self-clamping with mirrored
  neighbors to avoid a one-pixel color border.
- [ ] **Synchronize pipeline timing once** — Record all CUDA events, synchronize
  once at the end, and calculate elapsed times in a batch.
- [ ] **Use a 1D BLC launch** — The buffer is contiguous and does not require 2D
  indexing in the naive kernel.

## Tests and engineering

- [ ] **Extract benchmark helpers** — Deduplicate event, warm-up, loop, and
  bandwidth logic across performance tests.
- [ ] **Use representative gamma benchmark input** — Replace zero-filled input,
  which exercises only the fast linear sRGB branch.
- [ ] **Remove duplicate timing output** — Give `execute()` and `printSummary()`
  distinct reporting responsibilities.
- [ ] **Simplify synchronous RAW loading** — Use a normal `cudaMemcpy`, or add a
  reusable pinned-buffer pool when streaming is implemented.
- [ ] **Fix synthetic generator edge cases** — Handle `width == 1` and very small
  heights, preferably with `numpy.linspace`.

## Completed engineering work

- [x] CMake block discovery uses `CONFIGURE_DEPENDS`.
- [x] Build, run, test, configuration, and ownership contracts are documented.
- [x] Block factory declarations are centralized in `include/blocks.h`.
- [x] Host downloads use `std::unique_ptr<uint8_t[]>` and are exception-safe.
- [x] All four common Bayer patterns dispatch to compile-time specializations.
- [x] AWB full-scale clamping uses the correct bit-depth range.
- [x] Pipeline input consumption and non-owning output views are explicit.
- [x] `BENCH_ITERS` reuses buffers and avoids preservation copies.
- [x] Sensor configuration rejects invalid ranges, dimensions, and gains.
- [x] `FrameBuffer::allocate()` rejects unsafe layout changes.
- [x] A configurable 3x3 color-correction matrix is applied after demosaic.
