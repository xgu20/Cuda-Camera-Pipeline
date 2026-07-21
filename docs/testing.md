# Testing and benchmarking

The test suite covers configuration parsing, buffer ownership, ISP block
correctness, Bayer-pattern dispatch, and selected 4K performance benchmarks.
Most tests require a working CUDA-capable NVIDIA GPU and compatible driver.

## Build the tests

Tests are enabled by the normal project build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The test executable is `build/tests/isp_tests`.

## Run with CTest

Run the complete suite and print failures:

```bash
ctest --test-dir build --output-on-failure
```

Run tests in parallel:

```bash
ctest --test-dir build -j 8 --output-on-failure
```

Run tests whose registered names match a regular expression:

```bash
ctest --test-dir build -R Demosaic --output-on-failure
```

## Run with GoogleTest

The test binary supports standard GoogleTest options:

```bash
./build/tests/isp_tests
./build/tests/isp_tests --gtest_list_tests
./build/tests/isp_tests --gtest_filter='DemosaicTest.*'
./build/tests/isp_tests --gtest_filter='*Performance_4K*'
```

## Performance tests

Selected blocks include 4K benchmarks with warm-up iterations, repeated kernel
launches, average runtime, and estimated memory bandwidth. These tests are
useful for local comparisons but are not stable cross-machine performance
requirements. GPU model, clocks, driver, CUDA version, thermals, and concurrent
workloads all affect the result.

For end-to-end steady-state timing, run the application repeatedly on one frame:

```bash
BENCH_ITERS=100 ./build/libreisp data/example.raw output.png
```

The first iteration includes one-time buffer allocation. Later iterations reuse
pipeline buffers. The CLI prints per-stage and total timing.

`execute()` permits in-place blocks to consume the input. Benchmarks use this
zero-copy path. Use `executePreservingInput()` in application code when the
original input must remain unchanged.

## Troubleshooting CUDA test failures

Check that the CUDA runtime can access a compatible GPU before investigating a
large group of test failures. Typical environment errors include:

```text
CUDA driver version is insufficient for CUDA runtime version
no CUDA-capable device is detected
```

These messages indicate a driver/runtime or GPU-access problem rather than an
individual ISP assertion failure. Confirm the driver supports the installed
CUDA Toolkit and that the current environment exposes the GPU.

When reporting a performance or CUDA-specific failure, include:

- GPU model
- CUDA Toolkit version
- NVIDIA driver version
- CMake build type and `CMAKE_CUDA_ARCHITECTURES`
- Exact command and failing test name
