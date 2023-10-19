__kernel __attribute__((reqd_work_group_size(BLOCK_ROW_LENGTH, (COL_LENGTH / ELEMENT_PER_THREAD), 1))) 
void global_memory_test(__global float2* src, __global float2* dst){
    uint batch = get_group_id(2);    
    uint col_id = get_global_id(0);
    uint row_id = get_global_id(1);

    uint offset = batch * FFT_LENGTH + row_id * ROW_LENGTH + col_id;

    __private float2 R0, R1, R2, R3;


    {
        __global float2* lwIn = src + offset;
        
        R0 = lwIn[0];
        R1 = lwIn[STRIDE];
        R2 = lwIn[STRIDE * 2];
        R3 = lwIn[STRIDE * 3];

    }

    {
        __global float2* lwOut = dst + offset;
        lwOut[0] = R0;
        lwOut[STRIDE] = R1;
        lwOut[STRIDE * 2] = R2;
        lwOut[STRIDE * 3] = R3;
    }
}