__kernel __attribute__((always_inline))
void micro_pass_7(__private uint rw, __private uint lid, __global float2* src, __global float2* dest)
{
    __private float2 R0, R1, R2, R3, R4, R5, R6, R7, R8;
    __private uint2 lid_dim2 = (uint2)(lid % 81, lid / 81);
    __private uint2 mid = (uint2)(lid_dim2.x % 81, lid_dim2.x / 81);
    {
        __private uint offset;
        {
            __private uint2 group_id = (uint2)(get_group_id(0), get_group_id(1));
            __private uint index;
            {
                __private uint sub_fftseq_num = group_id.x * 3 + lid_dim2.y;
                __private uint2 sub_fftseq_num_dim2 = (uint2)(sub_fftseq_num / 729, sub_fftseq_num % (729));
                index = (sub_fftseq_num_dim2.x) * 531441 + lid_dim2.x * 729 + (sub_fftseq_num_dim2.y);
            }

            offset = group_id.y * 531441 + (index);
        }

        {
            __global float2* lwIn = src + offset;
            {
                R0 = lwIn[0];
                R1 = lwIn[59049];
                R2 = lwIn[118098];
                R3 = lwIn[177147];
                R4 = lwIn[236196];
                R5 = lwIn[295245];
                R6 = lwIn[354294];
                R7 = lwIn[413343];
                R8 = lwIn[472392];
            }

        }

    }

    mid = (uint2)(lid_dim2.x % 81, lid_dim2.x / 81);
    {
        __private uint offset;
        {
            __private uint2 group_id = (uint2)(get_group_id(0), get_group_id(1));
            __private uint index;
            {
                __private uint sub_fftseq_num = group_id.x * 3 + lid_dim2.y;
                __private uint2 sub_fftseq_num_dim2 = (uint2)(sub_fftseq_num / 729, sub_fftseq_num % (729));
                index = (sub_fftseq_num_dim2.x) * 531441 + lid_dim2.x * 729 + (sub_fftseq_num_dim2.y);
            }

            __private uint output_index = index;
            offset = group_id.y * 531441 + output_index;
        }

        {
            __global float2* lwOut = dest + offset;
            {
                lwOut[0] = R0;
                lwOut[59049] = R1;
                lwOut[118098] = R2;
                lwOut[177147] = R3;
                lwOut[236196] = R4;
                lwOut[295245] = R5;
                lwOut[354294] = R6;
                lwOut[413343] = R7;
                lwOut[472392] = R8;
            }

        }

    }

}

__kernel __attribute__((reqd_work_group_size(243, 1, 1)))
void fft_729_test(__global float2* input, __global float2* output)
{
    __private uint lid = get_local_id(0);
    __private uint rw = 1;
    __local float2 lm[2187];
    {
        {
                micro_pass_7(rw, lid, input, output);
        }

    }

}

