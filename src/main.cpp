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

    // fft_2401_test(device, queue, context);
    // fft_2401_test_no_twiddle(device, queue, context);
    // fft_2401_test_no_compute(device, queue, context);
    fft_2401_test_only_io(device, queue, context);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
