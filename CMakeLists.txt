cmake_minimum_required(VERSION 3.3.2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(FFTBenchmark)

find_package(OpenCL REQUIRED)

# 设定源代码文件和kernel文件的变量
set(SOURCE_FILES src/main.cpp src/fft_util.cpp src/util.cpp 
                 src/benchmark/ddr_test_1.cpp
                 src/benchmark/ddr_test_2.cpp
                 src/benchmark/l1_cacheline_test.cpp
                 )
                 
set(HEAD_FILES src/fft_util.hpp src/util.hpp src/benchmark/benchmark.hpp) 

# 创建可执行文件
add_executable(Benchmark ${SOURCE_FILES} ${HEAD_FILES})

# 链接 OpenCL 库
target_link_libraries(Benchmark PRIVATE ${OpenCL_LIBRARIES})

# 指定include目录
target_include_directories(Benchmark PRIVATE ${OpenCL_INCLUDE_DIRS})

target_include_directories(Benchmark PRIVATE ${PROJECT_SOURCE_DIR}/src)
