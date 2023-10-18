__kernel __attribute__((reqd_work_group_size(THREAD_PER_BLOCK, 1, 1))) 
void global_memory_pow7(__global float2* src, __global float2* dst){
    uint lid = get_local_id(0);
    uint2 lid_dim2 = (uint2)(lid % BLOCK_ROW_LENGTH, lid / BLOCK_ROW_LENGTH);

    uint2 gid = (uint2)(get_group_id(0), get_group_id(1));
    
    uint col_id = gid.x * BLOCK_ROW_LENGTH + lid_dim2.x;
    uint row_id = lid_dim2.y;
    uint offset = gid.y * FFT_LENGTH + row_id * ROW_LENGTH + col_id;

    __private float2 R0, R1, R2, R3, R4, R5, R6;


    {
        __global float2* lwIn = src + offset;
        R0 = lwIn[0];
        R1 = lwIn[STRIDE];
        R2 = lwIn[STRIDE * 2];
        R3 = lwIn[STRIDE * 3];
        R4 = lwIn[STRIDE * 4];
        R5 = lwIn[STRIDE * 5];
        R6 = lwIn[STRIDE * 6];
    }

    {
        __global float2* lwOut = dst + offset;
        lwOut[0] = R0;
        lwOut[STRIDE] = R1;
        lwOut[STRIDE * 2] = R2;
        lwOut[STRIDE * 3] = R3;
        lwOut[STRIDE * 4] = R4;
        lwOut[STRIDE * 5] = R5;
        lwOut[STRIDE * 6] = R6;
    }
}