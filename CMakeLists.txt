cmake_minimum_required(VERSION 3.25)
project(cuda_isp LANGUAGES CXX CUDA)

# ---- Default to Release build (zero optimization in empty build type kills
#      perf and disables NDEBUG, leaving asserts active). Override with
#      -DCMAKE_BUILD_TYPE=Debug for debugging.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- C++ / CUDA standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect GPU architecture of the build machine
set(CMAKE_CUDA_ARCHITECTURES native)

# Export compile commands for IDEs (clangd, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# ---- Find CUDA Toolkit ----
find_package(CUDAToolkit REQUIRED)

# ---- Collect sources ----
file(GLOB BLOCK_SOURCES "${CMAKE_SOURCE_DIR}/blocks/*.cu")

add_executable(cuda_isp
    src/main.cpp
    src/isp_pipeline.cpp
    src/frame_loader.cpp
    src/sensor_config.cpp
    ${BLOCK_SOURCES}
)

target_include_directories(cuda_isp PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# ---- JSON dependency (header-only) ----
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(json)

target_link_libraries(cuda_isp PRIVATE CUDA::cudart nlohmann_json::nlohmann_json)

# Enable separable compilation so .cu files can reference each other
set_target_properties(cuda_isp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ---- Testing ----
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)
