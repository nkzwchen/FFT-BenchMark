#include <CL/cl.h>
#include <stdio.h>

#include "fft_util.hpp"
#include "util.hpp"
#include "benchmark/benchmark.hpp"

int main() {

    // Init OpenCL
    cl_platform_id platform = InitPlatform();
    cl_device_id device = InitDevice(platform);
    cl_context context = CreateContext(device);
    cl_command_queue queue = CreateCommandQueue(context, device);
    printf("OpenCL Init Success\n");

    // test_global_memory_pow2(device, queue, context);
    // test_global_memory_pow3(device, queue, context);
    // test_global_memory_pow3_with_branch(device, queue, context);
    register_spill(device, queue, context);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
