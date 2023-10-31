cmake_minimum_required(VERSION 3.10)

project(FFTBenchmark)

find_package(OpenCL REQUIRED)

# 设定源代码文件和kernel文件的变量
set(SOURCE_FILES src/main.cpp src/fft_util.cpp src/util.cpp 
                 src/benchmark/global_memory_pow2.cpp 
                 src/benchmark/global_memory_pow3.cpp 
                 src/benchmark/global_memory_pow3_with_branch.cpp 
                 src/benchmark/global_memory_pow7.cpp
                 src/benchmark/register_spill.cpp
                 src/benchmark/fft_2401_test_only_io.cpp
                 src/benchmark/fft_2401_test_no_compute.cpp
                 src/benchmark/fft_2401_test_no_twiddle.cpp
                 src/benchmark/fft_2401_test.cpp
                 src/benchmark/fft_2401_read_by_col.cpp)
                 
set(HEAD_FILES src/fft_util.hpp src/util.hpp src/benchmark/benchmark.hpp) 

# 创建可执行文件
add_executable(Benchmark ${SOURCE_FILES} ${HEAD_FILES})

# 链接 OpenCL 库
target_link_libraries(Benchmark PRIVATE OpenCL::OpenCL)

# 指定include目录
target_include_directories(Benchmark PRIVATE ${OpenCL_INCLUDE_DIRS})

target_include_directories(Benchmark PRIVATE ${PROJECT_SOURCE_DIR}/src)
