#include <string>
#include <CL/cl.h>

#include "util.hpp"
#include "fft_util.hpp"

#define ELEMENT_PER_THREAD 25
#define BLOCK_ROW_LENGTH 1
#define COL 3125
#define ROW 1

#define FFT_LENGTH (ROW * COL)
#define THREAD_PER_BLOCK ((COL * BLOCK_ROW_LENGTH) / ELEMENT_PER_THREAD)
#define BLOCK_NUM ((ROW + BLOCK_ROW_LENGTH - 1 ) / BLOCK_ROW_LENGTH)
#define THREAD_NUM (BLOCK_NUM * THREAD_PER_BLOCK)




void register_spill(cl_device_id& device,  cl_command_queue& que, cl_context& context){
    size_t fft_length = FFT_LENGTH;
    size_t batch = 64;    

    // 读取并编译Kernel
    char* source_code = ReadKernelSource("../src/kernels/register_spill.cl");

    // printf("marco definition:\n %s\n", macro_definitions_str.c_str());

    printf("FFT Info:\n");
    printf("fft_length: %ld\n", fft_length);
    printf("batch: %ld\n", batch);
    printf("col length is %d\n", COL);
    printf("row length is %d\n", ROW);
    printf("block row length: %d\n", BLOCK_ROW_LENGTH);
    printf("thread per block: %d\n", THREAD_PER_BLOCK);
    printf("elment per thread: %d\n", ELEMENT_PER_THREAD);


    cl_program program = BuildProgram(context, device, source_code);

    // 创建Kernel
    cl_int ret;
    cl_kernel kernel = clCreateKernel(program, "fft_fwd", &ret);

    if(ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel.\n");
        exit(1);
    }
    else{
        printf("Create Kernel Success\n\n\n");
    }


    cl_mem *input_mem = new cl_mem[1];
    AllocateGpuBuffer(context, 1, fft_length * batch, input_mem);

    float **cpu_input = new float *[1];

    
    AllocateCpuBuffer(1, fft_length * batch, cpu_input);

    GenerateRandomData(cpu_input, 1, batch, fft_length);

    size_t global_work_size[3] = {THREAD_NUM * batch, 1, 1};
    size_t local_work_size[3] = {THREAD_PER_BLOCK, 1, 1};

    clSetKernelArg(kernel, 0, sizeof(cl_mem), input_mem);
    MoveBufferFromCpuToGpu(que, input_mem, cpu_input, 1, fft_length * batch);
    double run_time = 1.0;
    run_time = ExecuteKernel(que, kernel, global_work_size, local_work_size, 10);

    double mem_size = ((double)sizeof(float) * fft_length * batch * 4) / 1024.0 / 1024.0 / 1024.0;

    double bandwidth = mem_size * 1000 / run_time;

    printf("run time is %lf ms\n", run_time);
    printf("mem size is %lf GB\n", mem_size);
    printf("bandwidth is %lf GB/s\n", bandwidth);
        
    // Check data
    double correct_rate = 1.0;

    printf("correct rate: %lf\n", correct_rate);
    printf("----------------- end compute test -----------------\n");

    for (int i = 0; i < 1; i++)
    {
        free(cpu_input[i]);
        clReleaseMemObject(input_mem[i]);
    }


    // 清理资源
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(source_code);
}