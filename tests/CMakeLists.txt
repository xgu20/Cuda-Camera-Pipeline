file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "*.cu" "*.cpp")

add_executable(isp_tests
    ${TEST_SOURCES}
)

target_include_directories(isp_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# Link against the blocks we want to test
file(GLOB BLOCK_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/blocks/*.cu")
target_sources(isp_tests PRIVATE ${BLOCK_SOURCES})

target_link_libraries(isp_tests PRIVATE
    gtest_main
    CUDA::cudart
)

set_target_properties(isp_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

include(GoogleTest)
gtest_discover_tests(isp_tests)
