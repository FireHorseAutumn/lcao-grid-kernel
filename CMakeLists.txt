cmake_minimum_required(VERSION 3.20)

# Select a supported CUDA host compiler before project() enables CUDA.
# With GCC 16 or newer, prefer g++-15; otherwise enable the compiler override
# and use C++20 for CUDA, since newer libstdc++ headers require it.
find_program(_cxx_compiler NAMES $ENV{CXX} c++ g++)
if(_cxx_compiler)
    execute_process(COMMAND ${_cxx_compiler} -dumpversion
                    OUTPUT_VARIABLE _gcc_version OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
endif()
if(_gcc_version VERSION_GREATER_EQUAL 16)
    find_program(_gcc15 NAMES g++-15)
    if(_gcc15)
        set(CMAKE_CUDA_HOST_COMPILER "${_gcc15}")
    else()
        set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler -std=c++20")
    endif()
endif()

project(lcao_grid_kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
else()
    set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

enable_testing()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)

add_library(lcao_core STATIC src/model.cpp src/cpu_ref.cpp)
target_include_directories(lcao_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(lcao_core PRIVATE -Wall -Wextra)

# The gate: Y_lm orthonormality and closed forms must pass before anything
# downstream is built or timed.
add_executable(test_harmonics tests/test_harmonics.cpp)
target_link_libraries(test_harmonics PRIVATE lcao_core)
target_compile_options(test_harmonics PRIVATE -Wall -Wextra)
add_test(NAME harmonics COMMAND test_harmonics)

add_executable(test_cpu_density tests/test_cpu_density.cpp)
target_link_libraries(test_cpu_density PRIVATE lcao_core)
target_compile_options(test_cpu_density PRIVATE -Wall -Wextra)
add_test(NAME cpu_density COMMAND test_cpu_density)

# Hand-written fused/unfused kernels; no vendor library kernels in the
# measured path.
add_library(lcao_gpu STATIC src/grid_gpu.cu)
target_include_directories(lcao_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lcao_gpu PUBLIC lcao_core CUDA::cudart)
target_compile_definitions(lcao_gpu PRIVATE LCG_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

add_executable(test_gpu_density tests/test_gpu_density.cpp)
target_link_libraries(test_gpu_density PRIVATE lcao_gpu)
target_compile_options(test_gpu_density PRIVATE -Wall -Wextra)
add_test(NAME gpu_density COMMAND test_gpu_density)

add_executable(bench_grid benchmarks/bench_grid.cpp)
target_link_libraries(bench_grid PRIVATE lcao_gpu)
target_compile_options(bench_grid PRIVATE -Wall -Wextra)
target_compile_definitions(bench_grid PRIVATE
  LCG_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
