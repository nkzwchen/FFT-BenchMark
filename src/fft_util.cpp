#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <time.h>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <CL/cl.h>


double RelativeError(double a, double b)
{
    double max_num = std::max(abs(a), abs(b));

    if (max_num < 1e-5)
        return 0;

    return fabs(a - b) / max_num;
}

void GenerateRandomData(float **cpu_buffer, size_t buffer_num, size_t batch, size_t fft_length)
{
    srand(16384);

    size_t element_size = 1;

    if (buffer_num == 1)
        element_size *= 2;

    for (int b = 0; b < batch; b++)
        for (int i = 0; i < fft_length; i++)
        {
            for (int it_ = 0; it_ < buffer_num; it_++)
            {
                int k = b * fft_length + i;
                for (int jt_ = 0; jt_ < element_size; jt_++)
                    cpu_buffer[it_][element_size * k + jt_] = k; //(rand() % 11) / 10.0;
            }
        }
};

void AllocateCpuBuffer(size_t buffer_num, size_t buffer_size, float **cpu_buffer)
{
    size_t element_size = 1;

    if (buffer_num == 1)
        element_size *= 2;

    for (int it_ = 0; it_ < buffer_num; it_++)
    {
        cpu_buffer[it_] = (float *)malloc(buffer_size * element_size * sizeof(float));
        memset(cpu_buffer[it_], 0, buffer_size * element_size * sizeof(float));
    }
};

void AllocateGpuBuffer(cl_context context, size_t buffer_num, size_t buffer_size, cl_mem *gpu_buffer)
{
    size_t element_size = 1;

    if (buffer_num == 1)
        element_size *= 2;

    for (int it_ = 0; it_ < buffer_num; it_++)
    {
        gpu_buffer[it_] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         buffer_size * element_size * sizeof(float), nullptr, nullptr);
    }
};

void MoveBufferFromCpuToGpu(cl_command_queue queue, cl_mem *gpu_buffer, float **cpu_buffer,
                            size_t buffer_num, size_t buffer_size)
{
    size_t element_size = 1;

    if (buffer_num == 1)
        element_size *= 2;

    for (int it_ = 0; it_ < buffer_num; it_++)
    {
        clEnqueueWriteBuffer(queue, gpu_buffer[it_], CL_TRUE, 0,
                             buffer_size * element_size * sizeof(float), cpu_buffer[it_], 0,
                             nullptr, nullptr);
    }
};

void MoveBufferFromGpuToCpu(cl_command_queue queue, cl_mem *gpu_buffer, float **cpu_buffer,
                            size_t buffer_num, size_t buffer_size)
{
    size_t element_size = 1;

    if (buffer_num == 1)
        element_size *= 2;

    for (int it_ = 0; it_ < buffer_num; it_++)
    {
        clEnqueueReadBuffer(queue, gpu_buffer[it_], CL_TRUE, 0,
                            buffer_size * element_size * sizeof(float), cpu_buffer[it_], 0,
                            nullptr, nullptr);
    }
};


double CheckData(size_t buffer_num, size_t fft_length, float **output, float **ref_output)
{
    double correct_num = 0;
    size_t element_size = 1;

    if (buffer_num == 1)
        element_size *= 2;

    for (int i = 0; i < fft_length; i++)
    {
        for (int it_ = 0; it_ < buffer_num; it_++)
        {
            double max_error = 0;

            for (int jt_ = 0; jt_ < element_size; jt_++)
            {
                double error = RelativeError(output[it_][element_size * i + jt_],
                                             ref_output[it_][element_size * i + jt_]);
                max_error = std::max(max_error, error);
            }

            if (max_error < 0.01)
            {
                correct_num++;
            }
            else{
                printf("Error: %lf\n", max_error);
                printf("index is %d\n", i);
                for (int jt_ = 0; jt_ < element_size; jt_++)
                {
                    printf("for %d part, output is %lf, ref is %lf\n", jt_, output[it_][element_size * i + jt_],  ref_output[it_][element_size * i + jt_]);

                }

                return 0.0;
            }
        }
    }
    return correct_num / fft_length;
};