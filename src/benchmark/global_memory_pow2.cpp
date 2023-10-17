#include <string>
#include <CL/cl.h>

#include "util.hpp"
#include "fft_util.hpp"

#define ELEMENT_PER_THREAD 4
#define COL 64
#define ROW (65536)

#define COL_THREAD (COL / ELEMENT_PER_THREAD)
#define BLOCK_ROW_LENGTH 8



#define FFT_LENGTH (ROW * COL)
#define THREAD_PER_BLOCK ((COL * BLOCK_ROW_LENGTH) / ELEMENT_PER_THREAD)
#define BLOCK_NUM ((ROW + BLOCK_ROW_LENGTH - 1 ) / BLOCK_ROW_LENGTH)
#define THREAD_NUM (BLOCK_NUM * THREAD_PER_BLOCK)

#define STRIDE ((COL / ELEMENT_PER_THREAD) * ROW)




void test_global_memory_pow2(cl_device_id& device,  cl_command_queue& que, cl_context& context){
    size_t fft_length = FFT_LENGTH;
    size_t batch = 1;
    if (fft_length < 2048 * 2048)
        batch = 2048 * 2048 / fft_length;
    

    std::string macro_definitions_str =
    "#define ELEMENT_PER_THREAD " + std::to_string(ELEMENT_PER_THREAD) + "\n" +
    "#define BLOCK_ROW_LENGTH " + std::to_string(BLOCK_ROW_LENGTH) + "\n" +
    "#define BATCH " + std::to_string(batch) + "\n" +
    "#define COL_LENGTH " + std::to_string(COL) + "\n" +
    "#define ROW_LENGTH " + std::to_string(ROW) + "\n" +
    "#define FFT_LENGTH " + std::to_string(FFT_LENGTH) + "\n" +
    "#define THREAD_PER_BLOCK " + std::to_string(THREAD_PER_BLOCK) + "\n" +
    "#define THREAD_NUM " + std::to_string(THREAD_NUM) + "\n" +
    "#define BLOCK_NUM " + std::to_string(BLOCK_NUM) + "\n" +
    "#define STRIDE " + std::to_string(STRIDE) + "\n";

    // 读取并编译Kernel
    char* source_code = ReadKernelSource("../src/kernels/global_memory_pow2.cl", macro_definitions_str);

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
    cl_kernel kernel = clCreateKernel(program, "global_memory_pow2", &ret);

    if(ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel.\n");
        exit(1);
    }
    else{
        printf("Create Kernel Success\n\n\n");
    }



    cl_mem *input_mem = new cl_mem[1];
    cl_mem *output_mem = new cl_mem[1];
    AllocateGpuBuffer(context, 1, fft_length * batch, input_mem);
    AllocateGpuBuffer(context, 1, fft_length * batch, output_mem);

    float **cpu_input = new float *[1];
    float **cpu_output = new float *[1];
    float **ref_cpu_output = new float *[1];

    
    AllocateCpuBuffer(1, fft_length * batch, cpu_input);
    AllocateCpuBuffer(1, fft_length * batch, cpu_output);
    AllocateCpuBuffer(1, fft_length * batch, ref_cpu_output);

    GenerateRandomData(cpu_input, 1, batch, fft_length);

    size_t global_work_size[3] = {THREAD_NUM, batch, 1};
    size_t local_work_size[3] = {THREAD_PER_BLOCK, 1, 1};

    clSetKernelArg(kernel, 0, sizeof(cl_mem), input_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), output_mem);
    MoveBufferFromCpuToGpu(que, input_mem, cpu_input, 1, fft_length * batch);
    double run_time = ExecuteKernel(que, kernel, global_work_size, local_work_size, 1);
    MoveBufferFromGpuToCpu(que, output_mem, cpu_output, 1, fft_length * batch);

    double mem_size = ((double)sizeof(float) * fft_length * batch * 4) / 1024.0 / 1024.0 / 1024.0;

    double bandwidth = mem_size * 1000 / run_time;

    printf("run time is %lf ms\n", run_time);
    printf("mem size is %lf GB\n", mem_size);
    printf("bandwidth is %lf GB/s\n", bandwidth);
        
    // Check data
    double correct_rate = 1.0;
    correct_rate = CheckData(1, fft_length, cpu_output, cpu_input);

    printf("correct rate: %lf\n", correct_rate);
    printf("----------------- end compute test -----------------\n");

    for (int i = 0; i < 1; i++)
    {
        free(cpu_input[i]);
        clReleaseMemObject(input_mem[i]);
    }

    for (int i = 0; i < 1; i++)
    {
        free(cpu_output[i]);
        free(ref_cpu_output[i]);
        clReleaseMemObject(output_mem[i]);
    }

    // 清理资源
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(source_code);
}