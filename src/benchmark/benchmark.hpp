#pragma once
#include <CL/cl.h>
void test_global_memory_pow2(cl_device_id& device,  cl_command_queue& que, cl_context& context);
void test_global_memory_pow3(cl_device_id& device,  cl_command_queue& que, cl_context& context);
void test_global_memory_pow3_with_branch(cl_device_id& device,  cl_command_queue& que, cl_context& context);
void test_global_memory_pow7(cl_device_id& device,  cl_command_queue& que, cl_context& context);
void register_spill(cl_device_id& device,  cl_command_queue& que, cl_context& context);
void fft_2187_test(cl_device_id& device,  cl_command_queue& que, cl_context& context);
void fft_2187_test_no_compute(cl_device_id& device,  cl_command_queue& que, cl_context& context);
void fft_2187_test_no_twiddle(cl_device_id& device,  cl_command_queue& que, cl_context& context);