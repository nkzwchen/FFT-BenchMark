__kernel __attribute__((reqd_work_group_size(THREAD_PER_BLOCK, 1, 1))) 
void global_memory_pow2(__global float2* src, __global float2* dst){
    uint lid = get_local_id(0);
    uint2 lid_dim2 = (uint2)(lid % BLOCK_ROW_LENGTH, lid / BLOCK_ROW_LENGTH);

    uint2 gid = (uint2)(get_group_id(0), get_group_id(1));
    
    uint col_id = gid.x * BLOCK_ROW_LENGTH + lid_dim2.x;
    uint row_id = lid_dim2.y;
    uint offset = gid.y * FFT_LENGTH + row_id * ROW_LENGTH + col_id;

    #if ELEMENT_PER_THREAD == 16
    __private float2 R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15;
    #elif ELEMENT_PER_THREAD == 8
    __private float2 R0, R1, R2, R3, R4, R5, R6, R7;
    #elif ELEMENT_PER_THREAD == 4
    __private float2 R0, R1, R2, R3;
    #elif ELEMENT_PER_THREAD == 2
    __private float2 R0, R1;
    #else
    #error Unsupported ELEMENT_PER_THREAD value
    #endif



    {
        __global float2* lwIn = src + offset;
        
        R0 = lwIn[0];
        R1 = lwIn[STRIDE];

        #if ELEMENT_PER_THREAD >= 4
        R2 = lwIn[STRIDE * 2];
        R3 = lwIn[STRIDE * 3];
        #endif

        #if ELEMENT_PER_THREAD >= 8 
        R4 = lwIn[STRIDE * 4]; 
        R5 = lwIn[STRIDE * 5]; 
        R6 = lwIn[STRIDE * 6]; 
        R7 = lwIn[STRIDE * 7];
        #endif

        #if ELEMENT_PER_THREAD >= 16 
        R8 = lwIn[STRIDE * 8];
        R9 = lwIn[STRIDE * 9];
        R10 = lwIn[STRIDE * 10];
        R11 = lwIn[STRIDE * 11];
        R12 = lwIn[STRIDE * 12];
        R13 = lwIn[STRIDE * 13];
        R14 = lwIn[STRIDE * 14];
        R15 = lwIn[STRIDE * 15];
        #endif

    }

    {
        __global float2* lwOut = dst + offset;
        lwOut[0] = R0;
        lwOut[STRIDE] = R1;
        #if ELEMENT_PER_THREAD >= 4
        lwOut[STRIDE * 2] = R2;
        lwOut[STRIDE * 3] = R3;
        #endif
        #if ELEMENT_PER_THREAD >= 8
        lwOut[STRIDE * 4] = R4;
        lwOut[STRIDE * 5] = R5;
        lwOut[STRIDE * 6] = R6;
        lwOut[STRIDE * 7] = R7;
        #endif
        #if ELEMENT_PER_THREAD >= 16
        lwOut[STRIDE * 8] = R8;
        lwOut[STRIDE * 9] = R9;
        lwOut[STRIDE * 10] = R10;
        lwOut[STRIDE * 11] = R11;
        lwOut[STRIDE * 12] = R12;
        lwOut[STRIDE * 13] = R13;
        lwOut[STRIDE * 14] = R14;
        lwOut[STRIDE * 15] = R15;
        #endif
    }
}