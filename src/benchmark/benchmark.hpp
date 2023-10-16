#pragma once
#include <CL/cl.h>
void test_global_memory_pow2(cl_device_id& device,  cl_command_queue& que, cl_context& context);
void test_global_memory_pow3(cl_device_id& device,  cl_command_queue& que, cl_context& context);
void test_global_memory_pow3_with_branch(cl_device_id& device,  cl_command_queue& que, cl_context& context);
void register_spill(cl_device_id& device,  cl_command_queue& que, cl_context& context);
