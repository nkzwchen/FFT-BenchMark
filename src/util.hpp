#pragma once
#include <CL/cl.h>
#include <string>

cl_platform_id InitPlatform();

cl_device_id InitDevice(cl_platform_id platform_id);

cl_context CreateContext(cl_device_id device_id);

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device_id);

char* ReadKernelSource(const char* filename, std::string macro_str = "");

cl_program BuildProgram(cl_context context, cl_device_id device_id, const char* source_code);

double ExecuteKernel(cl_command_queue& que, cl_kernel& kernel, size_t* global_work_size, size_t* local_work_size, size_t test_cnt = 1);
void checkError(cl_int err, const char* operation);