#pragma once
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <time.h>
#include <cstdio>

#include <CL/cl.h>

double RelativeError(double a, double b);
void GenerateRandomData(float **cpu_buffer, size_t buffer_num, size_t batch, size_t fft_length);
void AllocateCpuBuffer(size_t buffer_num, size_t buffer_size, float **cpu_buffer);
void AllocateGpuBuffer(cl_context context, size_t buffer_num, size_t buffer_size, cl_mem *gpu_buffer);
void MoveBufferFromCpuToGpu(cl_command_queue queue, cl_mem *gpu_buffer, float **cpu_buffer,
                            size_t buffer_num, size_t buffer_size);
void MoveBufferFromGpuToCpu(cl_command_queue queue, cl_mem *gpu_buffer, float **cpu_buffer,
                            size_t buffer_num, size_t buffer_size);
double CheckData(size_t buffer_num, size_t fft_length, float **output, float **ref_output);