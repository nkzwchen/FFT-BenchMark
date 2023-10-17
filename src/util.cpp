#include <CL/cl.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <cstring>
#include <iostream>


#include <cmath>
#include <sys/time.h>
#include <time.h>
void checkError(cl_int err, const char* operation);

double jmGetTimeUs()
{
    unsigned long long time_us = 0;
    unsigned long long t = 1000000;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    time_us = (tv.tv_sec * t) + tv.tv_usec;
    return time_us;
};

cl_platform_id InitPlatform() {
    cl_platform_id platform_id;
    cl_int ret = clGetPlatformIDs(1, &platform_id, NULL);
    if(ret != CL_SUCCESS) {
        printf("Init PlatForm Error\n");
    }
    return platform_id;
}

cl_device_id InitDevice(cl_platform_id platform_id) {
    cl_device_id device_id;
    cl_int ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if(ret != CL_SUCCESS) {
        printf("Init Device Error\n");
    }
    return device_id;
}

cl_context CreateContext(cl_device_id device_id) {
    cl_context context;
    cl_int ret;
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if(ret != CL_SUCCESS) {
        printf("Init Context Error\n");
    }
    return context;
}


cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device_id) {
    cl_command_queue command_queue;
    cl_int ret;
    command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
    if(ret != CL_SUCCESS) {
        printf("Command Queue Error\n");
    }
    return command_queue;
}

char* ReadKernelSource(const char* filename, std::string macro_str) {
    FILE* file = fopen(filename, "r");
    if(!file) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    long file_length = ftell(file);
    long macro_length = macro_str.length();
    long length = file_length + macro_length;

    rewind(file);
    char* source_code = (char*)malloc(length + 1);
    memcpy(source_code, macro_str.c_str(), macro_length);
    fread(source_code + macro_length, sizeof(char), file_length, file);
    source_code[length] = '\0';
    fclose(file);
    return source_code;
}

cl_program BuildProgram(cl_context context, cl_device_id device_id, const char* source_code) {
    cl_int ret;
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_code, NULL, &ret);
    if(ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create CL program from source.\n");
        exit(1);
    }
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if(ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Error in kernel:\n%s\n", log);
        free(log);
        exit(1);
    }
    return program;
}

double ExecuteKernel(cl_command_queue& que, cl_kernel& kernel, size_t* global_work_size, size_t* local_work_size, size_t test_cnt){    
    // double wall_start = jmGetTimeUs();
    cl_ulong total_time = 0;

    for(int i = 0; i < test_cnt; i++){

        cl_event event;
        cl_int err = clEnqueueNDRangeKernel(que, kernel, 3, nullptr, global_work_size, local_work_size, 0, nullptr, &event);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to execute kernel!\n");
            checkError(err, "clEnqueueNDRangeKernel");
            return 0;
        }

        // 等待 Kernel 执行完成
        clWaitForEvents(1, &event);
        // double wall_end = jmGetTimeUs();

        cl_ulong time_start, time_end;
        

        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to get profiling start info!\n");
            checkError(err, "clGetEventProfilingInfo");
            return 0;
        }

        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to get profiling end info!\n");
            checkError(err, "clGetEventProfilingInfo");
            return 0;
        }

        total_time += time_end - time_start;
        double cur_time = (time_end - time_start) / 1000.0 / 1000.0;
        printf("run time is %lf ms\n", cur_time);
    }
    // double wall_time = wall_end - wall_start;
    double ret_time = total_time / 1000.0 / 1000.0 / test_cnt;

    return ret_time;

}


const char* clGetErrorString(cl_int error)
{
    switch(error)
    {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown";
    }
}

void checkError(cl_int err, const char* operation)
{
    if(err != CL_SUCCESS)
    {
        std::cerr << "Error during operation " << operation << ": " << clGetErrorString(err) << std::endl;
        exit(1);
    }
}