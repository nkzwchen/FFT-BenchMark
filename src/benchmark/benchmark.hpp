#pragma once
#include <CL/cl.h>
void ddr_test_1(cl_device_id& device,  cl_command_queue& que, cl_context& context);
void ddr_test_2(cl_device_id& device,  cl_command_queue& que, cl_context& context);
void l1_cacheline_test(cl_device_id& device,  cl_command_queue& que, cl_context& context);
