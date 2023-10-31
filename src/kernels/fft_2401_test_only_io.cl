#define DIRECTION 1

__kernel __attribute__((always_inline))
void micro_pass_0(__private uint rw, __private uint lid, __global float2* src, __local float2* dest)
{
    __private float2 R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46, R47, R48;
    __private uint2 lid_dim2 = (uint2)(lid % 49, lid / 49);
    __private uint2 mid = (uint2)(lid_dim2.x % 1, lid_dim2.x / 1);
    {
        __private uint offset;
        {
            __private uint2 group_id = (uint2)(get_group_id(0), get_group_id(1));
            __private uint index;
            {
                __private uint sub_fftseq_num = group_id.x * 1 + lid_dim2.y;
                __private uint2 sub_fftseq_num_dim2 = (uint2)(sub_fftseq_num / 1, sub_fftseq_num % (1));
                index = (sub_fftseq_num_dim2.x) * 2401 + lid_dim2.x * 1 + (sub_fftseq_num_dim2.y);
            }

            offset = group_id.y * 2401 + (index);
        }

        {
            __global float2* lwIn = src + offset;
            {
                R0 = lwIn[0];
                R1 = lwIn[343];
                R2 = lwIn[686];
                R3 = lwIn[1029];
                R4 = lwIn[1372];
                R5 = lwIn[1715];
                R6 = lwIn[2058];
            }

            {
                R7 = lwIn[49];
                R8 = lwIn[392];
                R9 = lwIn[735];
                R10 = lwIn[1078];
                R11 = lwIn[1421];
                R12 = lwIn[1764];
                R13 = lwIn[2107];
            }

            {
                R14 = lwIn[98];
                R15 = lwIn[441];
                R16 = lwIn[784];
                R17 = lwIn[1127];
                R18 = lwIn[1470];
                R19 = lwIn[1813];
                R20 = lwIn[2156];
            }

            {
                R21 = lwIn[147];
                R22 = lwIn[490];
                R23 = lwIn[833];
                R24 = lwIn[1176];
                R25 = lwIn[1519];
                R26 = lwIn[1862];
                R27 = lwIn[2205];
            }

            {
                R28 = lwIn[196];
                R29 = lwIn[539];
                R30 = lwIn[882];
                R31 = lwIn[1225];
                R32 = lwIn[1568];
                R33 = lwIn[1911];
                R34 = lwIn[2254];
            }

            {
                R35 = lwIn[245];
                R36 = lwIn[588];
                R37 = lwIn[931];
                R38 = lwIn[1274];
                R39 = lwIn[1617];
                R40 = lwIn[1960];
                R41 = lwIn[2303];
            }

            {
                R42 = lwIn[294];
                R43 = lwIn[637];
                R44 = lwIn[980];
                R45 = lwIn[1323];
                R46 = lwIn[1666];
                R47 = lwIn[2009];
                R48 = lwIn[2352];
            }

        }

    }

    {
        __private uint offset;
        {
            __private uint2 element_id = (uint2)(lid_dim2.y, mid.y * 7 + mid.x);
            offset = element_id.y * 1 +  element_id.x * 2401;
        }

        {
            {
                dest[offset + 0] = R0;
                dest[offset + 1] = R1;
                dest[offset + 2] = R2;
                dest[offset + 3] = R3;
                dest[offset + 4] = R4;
                dest[offset + 5] = R5;
                dest[offset + 6] = R6;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 343] = R7;
                dest[offset + 344] = R8;
                dest[offset + 345] = R9;
                dest[offset + 346] = R10;
                dest[offset + 347] = R11;
                dest[offset + 348] = R12;
                dest[offset + 349] = R13;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 686] = R14;
                dest[offset + 687] = R15;
                dest[offset + 688] = R16;
                dest[offset + 689] = R17;
                dest[offset + 690] = R18;
                dest[offset + 691] = R19;
                dest[offset + 692] = R20;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 1029] = R21;
                dest[offset + 1030] = R22;
                dest[offset + 1031] = R23;
                dest[offset + 1032] = R24;
                dest[offset + 1033] = R25;
                dest[offset + 1034] = R26;
                dest[offset + 1035] = R27;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 1372] = R28;
                dest[offset + 1373] = R29;
                dest[offset + 1374] = R30;
                dest[offset + 1375] = R31;
                dest[offset + 1376] = R32;
                dest[offset + 1377] = R33;
                dest[offset + 1378] = R34;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 1715] = R35;
                dest[offset + 1716] = R36;
                dest[offset + 1717] = R37;
                dest[offset + 1718] = R38;
                dest[offset + 1719] = R39;
                dest[offset + 1720] = R40;
                dest[offset + 1721] = R41;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 2058] = R42;
                dest[offset + 2059] = R43;
                dest[offset + 2060] = R44;
                dest[offset + 2061] = R45;
                dest[offset + 2062] = R46;
                dest[offset + 2063] = R47;
                dest[offset + 2064] = R48;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

    }

}

__kernel __attribute__((always_inline))
void micro_pass_1(__private uint rw, __private uint lid, __local float2* src, __local float2* dest)
{
    __private float2 R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46, R47, R48;
    __private uint2 lid_dim2 = (uint2)(lid % 49, lid / 49);
    __private uint2 mid = (uint2)(lid_dim2.x % 7, lid_dim2.x / 7);
    {
        __private uint offset = lid_dim2.x * 1 + lid_dim2.y * 2401;
        {
            {
                R0 = src[offset + 0];
                R1 = src[offset + 343];
                R2 = src[offset + 686];
                R3 = src[offset + 1029];
                R4 = src[offset + 1372];
                R5 = src[offset + 1715];
                R6 = src[offset + 2058];
            }

            {
                R7 = src[offset + 49];
                R8 = src[offset + 392];
                R9 = src[offset + 735];
                R10 = src[offset + 1078];
                R11 = src[offset + 1421];
                R12 = src[offset + 1764];
                R13 = src[offset + 2107];
            }

            {
                R14 = src[offset + 98];
                R15 = src[offset + 441];
                R16 = src[offset + 784];
                R17 = src[offset + 1127];
                R18 = src[offset + 1470];
                R19 = src[offset + 1813];
                R20 = src[offset + 2156];
            }

            {
                R21 = src[offset + 147];
                R22 = src[offset + 490];
                R23 = src[offset + 833];
                R24 = src[offset + 1176];
                R25 = src[offset + 1519];
                R26 = src[offset + 1862];
                R27 = src[offset + 2205];
            }

            {
                R28 = src[offset + 196];
                R29 = src[offset + 539];
                R30 = src[offset + 882];
                R31 = src[offset + 1225];
                R32 = src[offset + 1568];
                R33 = src[offset + 1911];
                R34 = src[offset + 2254];
            }

            {
                R35 = src[offset + 245];
                R36 = src[offset + 588];
                R37 = src[offset + 931];
                R38 = src[offset + 1274];
                R39 = src[offset + 1617];
                R40 = src[offset + 1960];
                R41 = src[offset + 2303];
            }

            {
                R42 = src[offset + 294];
                R43 = src[offset + 637];
                R44 = src[offset + 980];
                R45 = src[offset + 1323];
                R46 = src[offset + 1666];
                R47 = src[offset + 2009];
                R48 = src[offset + 2352];
            }

        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    {
        __private uint offset;
        {
            __private uint2 element_id = (uint2)(lid_dim2.y, mid.y * 49 + mid.x);
            offset = element_id.y * 1 +  element_id.x * 2401;
        }

        {
            {
                dest[offset + 0] = R0;
                dest[offset + 7] = R1;
                dest[offset + 14] = R2;
                dest[offset + 21] = R3;
                dest[offset + 28] = R4;
                dest[offset + 35] = R5;
                dest[offset + 42] = R6;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 343] = R7;
                dest[offset + 350] = R8;
                dest[offset + 357] = R9;
                dest[offset + 364] = R10;
                dest[offset + 371] = R11;
                dest[offset + 378] = R12;
                dest[offset + 385] = R13;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 686] = R14;
                dest[offset + 693] = R15;
                dest[offset + 700] = R16;
                dest[offset + 707] = R17;
                dest[offset + 714] = R18;
                dest[offset + 721] = R19;
                dest[offset + 728] = R20;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 1029] = R21;
                dest[offset + 1036] = R22;
                dest[offset + 1043] = R23;
                dest[offset + 1050] = R24;
                dest[offset + 1057] = R25;
                dest[offset + 1064] = R26;
                dest[offset + 1071] = R27;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 1372] = R28;
                dest[offset + 1379] = R29;
                dest[offset + 1386] = R30;
                dest[offset + 1393] = R31;
                dest[offset + 1400] = R32;
                dest[offset + 1407] = R33;
                dest[offset + 1414] = R34;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 1715] = R35;
                dest[offset + 1722] = R36;
                dest[offset + 1729] = R37;
                dest[offset + 1736] = R38;
                dest[offset + 1743] = R39;
                dest[offset + 1750] = R40;
                dest[offset + 1757] = R41;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 2058] = R42;
                dest[offset + 2065] = R43;
                dest[offset + 2072] = R44;
                dest[offset + 2079] = R45;
                dest[offset + 2086] = R46;
                dest[offset + 2093] = R47;
                dest[offset + 2100] = R48;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

    }

}

__kernel __attribute__((always_inline))
void micro_pass_2(__private uint rw, __private uint lid, __local float2* src, __local float2* dest)
{
    __private float2 R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46, R47, R48;
    __private uint2 lid_dim2 = (uint2)(lid % 49, lid / 49);
    __private uint2 mid = (uint2)(lid_dim2.x % 49, lid_dim2.x / 49);
    {
        __private uint offset = lid_dim2.x * 1 + lid_dim2.y * 2401;
        {
            {
                R0 = src[offset + 0];
                R1 = src[offset + 343];
                R2 = src[offset + 686];
                R3 = src[offset + 1029];
                R4 = src[offset + 1372];
                R5 = src[offset + 1715];
                R6 = src[offset + 2058];
            }

            {
                R7 = src[offset + 49];
                R8 = src[offset + 392];
                R9 = src[offset + 735];
                R10 = src[offset + 1078];
                R11 = src[offset + 1421];
                R12 = src[offset + 1764];
                R13 = src[offset + 2107];
            }

            {
                R14 = src[offset + 98];
                R15 = src[offset + 441];
                R16 = src[offset + 784];
                R17 = src[offset + 1127];
                R18 = src[offset + 1470];
                R19 = src[offset + 1813];
                R20 = src[offset + 2156];
            }

            {
                R21 = src[offset + 147];
                R22 = src[offset + 490];
                R23 = src[offset + 833];
                R24 = src[offset + 1176];
                R25 = src[offset + 1519];
                R26 = src[offset + 1862];
                R27 = src[offset + 2205];
            }

            {
                R28 = src[offset + 196];
                R29 = src[offset + 539];
                R30 = src[offset + 882];
                R31 = src[offset + 1225];
                R32 = src[offset + 1568];
                R33 = src[offset + 1911];
                R34 = src[offset + 2254];
            }

            {
                R35 = src[offset + 245];
                R36 = src[offset + 588];
                R37 = src[offset + 931];
                R38 = src[offset + 1274];
                R39 = src[offset + 1617];
                R40 = src[offset + 1960];
                R41 = src[offset + 2303];
            }

            {
                R42 = src[offset + 294];
                R43 = src[offset + 637];
                R44 = src[offset + 980];
                R45 = src[offset + 1323];
                R46 = src[offset + 1666];
                R47 = src[offset + 2009];
                R48 = src[offset + 2352];
            }

        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    {
        __private uint offset;
        {
            __private uint2 element_id = (uint2)(lid_dim2.y, mid.y * 343 + mid.x);
            offset = element_id.y * 1 +  element_id.x * 2401;
        }

        {
            {
                dest[offset + 0] = R0;
                dest[offset + 49] = R1;
                dest[offset + 98] = R2;
                dest[offset + 147] = R3;
                dest[offset + 196] = R4;
                dest[offset + 245] = R5;
                dest[offset + 294] = R6;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 343] = R7;
                dest[offset + 392] = R8;
                dest[offset + 441] = R9;
                dest[offset + 490] = R10;
                dest[offset + 539] = R11;
                dest[offset + 588] = R12;
                dest[offset + 637] = R13;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 686] = R14;
                dest[offset + 735] = R15;
                dest[offset + 784] = R16;
                dest[offset + 833] = R17;
                dest[offset + 882] = R18;
                dest[offset + 931] = R19;
                dest[offset + 980] = R20;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 1029] = R21;
                dest[offset + 1078] = R22;
                dest[offset + 1127] = R23;
                dest[offset + 1176] = R24;
                dest[offset + 1225] = R25;
                dest[offset + 1274] = R26;
                dest[offset + 1323] = R27;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 1372] = R28;
                dest[offset + 1421] = R29;
                dest[offset + 1470] = R30;
                dest[offset + 1519] = R31;
                dest[offset + 1568] = R32;
                dest[offset + 1617] = R33;
                dest[offset + 1666] = R34;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 1715] = R35;
                dest[offset + 1764] = R36;
                dest[offset + 1813] = R37;
                dest[offset + 1862] = R38;
                dest[offset + 1911] = R39;
                dest[offset + 1960] = R40;
                dest[offset + 2009] = R41;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                dest[offset + 2058] = R42;
                dest[offset + 2107] = R43;
                dest[offset + 2156] = R44;
                dest[offset + 2205] = R45;
                dest[offset + 2254] = R46;
                dest[offset + 2303] = R47;
                dest[offset + 2352] = R48;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

    }

}

__kernel __attribute__((always_inline))
void micro_pass_3(__private uint rw, __private uint lid, __local float2* src, __global float2* dest)
{
    __private float2 R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46, R47, R48;
    __private uint2 lid_dim2 = (uint2)(lid % 49, lid / 49);
    __private uint2 mid = (uint2)(lid_dim2.x % 343, lid_dim2.x / 343);
    {
        __private uint offset = lid_dim2.x * 1 + lid_dim2.y * 2401;
        {
            {
                R0 = src[offset + 0];
                R1 = src[offset + 343];
                R2 = src[offset + 686];
                R3 = src[offset + 1029];
                R4 = src[offset + 1372];
                R5 = src[offset + 1715];
                R6 = src[offset + 2058];
            }

            {
                R7 = src[offset + 49];
                R8 = src[offset + 392];
                R9 = src[offset + 735];
                R10 = src[offset + 1078];
                R11 = src[offset + 1421];
                R12 = src[offset + 1764];
                R13 = src[offset + 2107];
            }

            {
                R14 = src[offset + 98];
                R15 = src[offset + 441];
                R16 = src[offset + 784];
                R17 = src[offset + 1127];
                R18 = src[offset + 1470];
                R19 = src[offset + 1813];
                R20 = src[offset + 2156];
            }

            {
                R21 = src[offset + 147];
                R22 = src[offset + 490];
                R23 = src[offset + 833];
                R24 = src[offset + 1176];
                R25 = src[offset + 1519];
                R26 = src[offset + 1862];
                R27 = src[offset + 2205];
            }

            {
                R28 = src[offset + 196];
                R29 = src[offset + 539];
                R30 = src[offset + 882];
                R31 = src[offset + 1225];
                R32 = src[offset + 1568];
                R33 = src[offset + 1911];
                R34 = src[offset + 2254];
            }

            {
                R35 = src[offset + 245];
                R36 = src[offset + 588];
                R37 = src[offset + 931];
                R38 = src[offset + 1274];
                R39 = src[offset + 1617];
                R40 = src[offset + 1960];
                R41 = src[offset + 2303];
            }

            {
                R42 = src[offset + 294];
                R43 = src[offset + 637];
                R44 = src[offset + 980];
                R45 = src[offset + 1323];
                R46 = src[offset + 1666];
                R47 = src[offset + 2009];
                R48 = src[offset + 2352];
            }

        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    {
        __private uint offset;
        {
            __private uint2 group_id = (uint2)(get_group_id(0), get_group_id(1));
            __private uint index;
            {
                __private uint sub_fftseq_num = group_id.x * 1 + lid_dim2.y;
                __private uint2 sub_fftseq_num_dim2 = (uint2)(sub_fftseq_num / 1, sub_fftseq_num % (1));
                index = (sub_fftseq_num_dim2.x) * 2401 + lid_dim2.x * 1 + (sub_fftseq_num_dim2.y);
            }

            __private uint output_index = index;
            offset = group_id.y * 2401 + output_index;
        }

        {
            __global float2* lwOut = dest + offset;
            {
                lwOut[0] = R0;
                lwOut[343] = R1;
                lwOut[686] = R2;
                lwOut[1029] = R3;
                lwOut[1372] = R4;
                lwOut[1715] = R5;
                lwOut[2058] = R6;
            }

            {
                lwOut[49] = R7;
                lwOut[392] = R8;
                lwOut[735] = R9;
                lwOut[1078] = R10;
                lwOut[1421] = R11;
                lwOut[1764] = R12;
                lwOut[2107] = R13;
            }

            {
                lwOut[98] = R14;
                lwOut[441] = R15;
                lwOut[784] = R16;
                lwOut[1127] = R17;
                lwOut[1470] = R18;
                lwOut[1813] = R19;
                lwOut[2156] = R20;
            }

            {
                lwOut[147] = R21;
                lwOut[490] = R22;
                lwOut[833] = R23;
                lwOut[1176] = R24;
                lwOut[1519] = R25;
                lwOut[1862] = R26;
                lwOut[2205] = R27;
            }

            {
                lwOut[196] = R28;
                lwOut[539] = R29;
                lwOut[882] = R30;
                lwOut[1225] = R31;
                lwOut[1568] = R32;
                lwOut[1911] = R33;
                lwOut[2254] = R34;
            }

            {
                lwOut[245] = R35;
                lwOut[588] = R36;
                lwOut[931] = R37;
                lwOut[1274] = R38;
                lwOut[1617] = R39;
                lwOut[1960] = R40;
                lwOut[2303] = R41;
            }

            {
                lwOut[294] = R42;
                lwOut[637] = R43;
                lwOut[980] = R44;
                lwOut[1323] = R45;
                lwOut[1666] = R46;
                lwOut[2009] = R47;
                lwOut[2352] = R48;
            }

        }

    }

}

__kernel __attribute__((reqd_work_group_size(49, 1, 1)))
void fft_2401_test_only_io(__global float2* input, __global float2* output)
{
    __private uint lid = get_local_id(0);
    __private uint rw = 1;
    __local float2 lm[2401];
    {
        {
                micro_pass_0(rw, lid, input, lm);
        }

    }

    {
        {
                micro_pass_1(rw, lid, lm, lm);
        }

    }

    {
        {
                micro_pass_2(rw, lid, lm, lm);
        }

    }

    {
        {
                micro_pass_3(rw, lid, lm, output);
        }

    }

}


