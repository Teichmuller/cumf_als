/*
 * reg_ops.h
 *
 * This is file is automatically generated. Please do not modify unless you know what you are doing.
 *
 *  Created on: Jul 19, 2017
 *      Author: Xuan
 */

#ifndef REG_OPS_H_
#define REG_OPS_H_

#define RegisterTileSize 10

#define declare_registers()\
    dtype temp0 = 0, temp1 = 0, temp2 = 0, temp3 = 0, temp4 = 0, temp5 = 0, temp6 = 0, temp7 = 0, temp8 = 0, temp9 = 0;\
    dtype temp10 = 0, temp11 = 0, temp12 = 0, temp13 = 0, temp14 = 0, temp15 = 0, temp16 = 0, temp17 = 0, temp18 = 0, temp19 = 0;\
    dtype temp20 = 0, temp21 = 0, temp22 = 0, temp23 = 0, temp24 = 0, temp25 = 0, temp26 = 0, temp27 = 0, temp28 = 0, temp29 = 0;\
    dtype temp30 = 0, temp31 = 0, temp32 = 0, temp33 = 0, temp34 = 0, temp35 = 0, temp36 = 0, temp37 = 0, temp38 = 0, temp39 = 0;\
    dtype temp40 = 0, temp41 = 0, temp42 = 0, temp43 = 0, temp44 = 0, temp45 = 0, temp46 = 0, temp47 = 0, temp48 = 0, temp49 = 0;\
    dtype temp50 = 0, temp51 = 0, temp52 = 0, temp53 = 0, temp54 = 0, temp55 = 0, temp56 = 0, temp57 = 0, temp58 = 0, temp59 = 0;\
    dtype temp60 = 0, temp61 = 0, temp62 = 0, temp63 = 0, temp64 = 0, temp65 = 0, temp66 = 0, temp67 = 0, temp68 = 0, temp69 = 0;\
    dtype temp70 = 0, temp71 = 0, temp72 = 0, temp73 = 0, temp74 = 0, temp75 = 0, temp76 = 0, temp77 = 0, temp78 = 0, temp79 = 0;\
    dtype temp80 = 0, temp81 = 0, temp82 = 0, temp83 = 0, temp84 = 0, temp85 = 0, temp86 = 0, temp87 = 0, temp88 = 0, temp89 = 0;\
    dtype temp90 = 0, temp91 = 0, temp92 = 0, temp93 = 0, temp94 = 0, temp95 = 0, temp96 = 0, temp97 = 0, temp98 = 0, temp99 = 0

#define accumulate_in_registers()\
    do\
    {\
        temp0 += thetaTemp[tile_x / 2 + 0 + k * f / 2].x * thetaTemp[tile_y / 2 + 0 + k * f / 2].x;\
        temp1 += thetaTemp[tile_x / 2 + 0 + k * f / 2].x * thetaTemp[tile_y / 2 + 0 + k * f / 2].y;\
        temp2 += thetaTemp[tile_x / 2 + 0 + k * f / 2].x * thetaTemp[tile_y / 2 + 1 + k * f / 2].x;\
        temp3 += thetaTemp[tile_x / 2 + 0 + k * f / 2].x * thetaTemp[tile_y / 2 + 1 + k * f / 2].y;\
        temp4 += thetaTemp[tile_x / 2 + 0 + k * f / 2].x * thetaTemp[tile_y / 2 + 2 + k * f / 2].x;\
        temp5 += thetaTemp[tile_x / 2 + 0 + k * f / 2].x * thetaTemp[tile_y / 2 + 2 + k * f / 2].y;\
        temp6 += thetaTemp[tile_x / 2 + 0 + k * f / 2].x * thetaTemp[tile_y / 2 + 3 + k * f / 2].x;\
        temp7 += thetaTemp[tile_x / 2 + 0 + k * f / 2].x * thetaTemp[tile_y / 2 + 3 + k * f / 2].y;\
        temp8 += thetaTemp[tile_x / 2 + 0 + k * f / 2].x * thetaTemp[tile_y / 2 + 4 + k * f / 2].x;\
        temp9 += thetaTemp[tile_x / 2 + 0 + k * f / 2].x * thetaTemp[tile_y / 2 + 4 + k * f / 2].y;\
        temp10 += thetaTemp[tile_x / 2 + 0 + k * f / 2].y * thetaTemp[tile_y / 2 + 0 + k * f / 2].x;\
        temp11 += thetaTemp[tile_x / 2 + 0 + k * f / 2].y * thetaTemp[tile_y / 2 + 0 + k * f / 2].y;\
        temp12 += thetaTemp[tile_x / 2 + 0 + k * f / 2].y * thetaTemp[tile_y / 2 + 1 + k * f / 2].x;\
        temp13 += thetaTemp[tile_x / 2 + 0 + k * f / 2].y * thetaTemp[tile_y / 2 + 1 + k * f / 2].y;\
        temp14 += thetaTemp[tile_x / 2 + 0 + k * f / 2].y * thetaTemp[tile_y / 2 + 2 + k * f / 2].x;\
        temp15 += thetaTemp[tile_x / 2 + 0 + k * f / 2].y * thetaTemp[tile_y / 2 + 2 + k * f / 2].y;\
        temp16 += thetaTemp[tile_x / 2 + 0 + k * f / 2].y * thetaTemp[tile_y / 2 + 3 + k * f / 2].x;\
        temp17 += thetaTemp[tile_x / 2 + 0 + k * f / 2].y * thetaTemp[tile_y / 2 + 3 + k * f / 2].y;\
        temp18 += thetaTemp[tile_x / 2 + 0 + k * f / 2].y * thetaTemp[tile_y / 2 + 4 + k * f / 2].x;\
        temp19 += thetaTemp[tile_x / 2 + 0 + k * f / 2].y * thetaTemp[tile_y / 2 + 4 + k * f / 2].y;\
        temp20 += thetaTemp[tile_x / 2 + 1 + k * f / 2].x * thetaTemp[tile_y / 2 + 0 + k * f / 2].x;\
        temp21 += thetaTemp[tile_x / 2 + 1 + k * f / 2].x * thetaTemp[tile_y / 2 + 0 + k * f / 2].y;\
        temp22 += thetaTemp[tile_x / 2 + 1 + k * f / 2].x * thetaTemp[tile_y / 2 + 1 + k * f / 2].x;\
        temp23 += thetaTemp[tile_x / 2 + 1 + k * f / 2].x * thetaTemp[tile_y / 2 + 1 + k * f / 2].y;\
        temp24 += thetaTemp[tile_x / 2 + 1 + k * f / 2].x * thetaTemp[tile_y / 2 + 2 + k * f / 2].x;\
        temp25 += thetaTemp[tile_x / 2 + 1 + k * f / 2].x * thetaTemp[tile_y / 2 + 2 + k * f / 2].y;\
        temp26 += thetaTemp[tile_x / 2 + 1 + k * f / 2].x * thetaTemp[tile_y / 2 + 3 + k * f / 2].x;\
        temp27 += thetaTemp[tile_x / 2 + 1 + k * f / 2].x * thetaTemp[tile_y / 2 + 3 + k * f / 2].y;\
        temp28 += thetaTemp[tile_x / 2 + 1 + k * f / 2].x * thetaTemp[tile_y / 2 + 4 + k * f / 2].x;\
        temp29 += thetaTemp[tile_x / 2 + 1 + k * f / 2].x * thetaTemp[tile_y / 2 + 4 + k * f / 2].y;\
        temp30 += thetaTemp[tile_x / 2 + 1 + k * f / 2].y * thetaTemp[tile_y / 2 + 0 + k * f / 2].x;\
        temp31 += thetaTemp[tile_x / 2 + 1 + k * f / 2].y * thetaTemp[tile_y / 2 + 0 + k * f / 2].y;\
        temp32 += thetaTemp[tile_x / 2 + 1 + k * f / 2].y * thetaTemp[tile_y / 2 + 1 + k * f / 2].x;\
        temp33 += thetaTemp[tile_x / 2 + 1 + k * f / 2].y * thetaTemp[tile_y / 2 + 1 + k * f / 2].y;\
        temp34 += thetaTemp[tile_x / 2 + 1 + k * f / 2].y * thetaTemp[tile_y / 2 + 2 + k * f / 2].x;\
        temp35 += thetaTemp[tile_x / 2 + 1 + k * f / 2].y * thetaTemp[tile_y / 2 + 2 + k * f / 2].y;\
        temp36 += thetaTemp[tile_x / 2 + 1 + k * f / 2].y * thetaTemp[tile_y / 2 + 3 + k * f / 2].x;\
        temp37 += thetaTemp[tile_x / 2 + 1 + k * f / 2].y * thetaTemp[tile_y / 2 + 3 + k * f / 2].y;\
        temp38 += thetaTemp[tile_x / 2 + 1 + k * f / 2].y * thetaTemp[tile_y / 2 + 4 + k * f / 2].x;\
        temp39 += thetaTemp[tile_x / 2 + 1 + k * f / 2].y * thetaTemp[tile_y / 2 + 4 + k * f / 2].y;\
        temp40 += thetaTemp[tile_x / 2 + 2 + k * f / 2].x * thetaTemp[tile_y / 2 + 0 + k * f / 2].x;\
        temp41 += thetaTemp[tile_x / 2 + 2 + k * f / 2].x * thetaTemp[tile_y / 2 + 0 + k * f / 2].y;\
        temp42 += thetaTemp[tile_x / 2 + 2 + k * f / 2].x * thetaTemp[tile_y / 2 + 1 + k * f / 2].x;\
        temp43 += thetaTemp[tile_x / 2 + 2 + k * f / 2].x * thetaTemp[tile_y / 2 + 1 + k * f / 2].y;\
        temp44 += thetaTemp[tile_x / 2 + 2 + k * f / 2].x * thetaTemp[tile_y / 2 + 2 + k * f / 2].x;\
        temp45 += thetaTemp[tile_x / 2 + 2 + k * f / 2].x * thetaTemp[tile_y / 2 + 2 + k * f / 2].y;\
        temp46 += thetaTemp[tile_x / 2 + 2 + k * f / 2].x * thetaTemp[tile_y / 2 + 3 + k * f / 2].x;\
        temp47 += thetaTemp[tile_x / 2 + 2 + k * f / 2].x * thetaTemp[tile_y / 2 + 3 + k * f / 2].y;\
        temp48 += thetaTemp[tile_x / 2 + 2 + k * f / 2].x * thetaTemp[tile_y / 2 + 4 + k * f / 2].x;\
        temp49 += thetaTemp[tile_x / 2 + 2 + k * f / 2].x * thetaTemp[tile_y / 2 + 4 + k * f / 2].y;\
        temp50 += thetaTemp[tile_x / 2 + 2 + k * f / 2].y * thetaTemp[tile_y / 2 + 0 + k * f / 2].x;\
        temp51 += thetaTemp[tile_x / 2 + 2 + k * f / 2].y * thetaTemp[tile_y / 2 + 0 + k * f / 2].y;\
        temp52 += thetaTemp[tile_x / 2 + 2 + k * f / 2].y * thetaTemp[tile_y / 2 + 1 + k * f / 2].x;\
        temp53 += thetaTemp[tile_x / 2 + 2 + k * f / 2].y * thetaTemp[tile_y / 2 + 1 + k * f / 2].y;\
        temp54 += thetaTemp[tile_x / 2 + 2 + k * f / 2].y * thetaTemp[tile_y / 2 + 2 + k * f / 2].x;\
        temp55 += thetaTemp[tile_x / 2 + 2 + k * f / 2].y * thetaTemp[tile_y / 2 + 2 + k * f / 2].y;\
        temp56 += thetaTemp[tile_x / 2 + 2 + k * f / 2].y * thetaTemp[tile_y / 2 + 3 + k * f / 2].x;\
        temp57 += thetaTemp[tile_x / 2 + 2 + k * f / 2].y * thetaTemp[tile_y / 2 + 3 + k * f / 2].y;\
        temp58 += thetaTemp[tile_x / 2 + 2 + k * f / 2].y * thetaTemp[tile_y / 2 + 4 + k * f / 2].x;\
        temp59 += thetaTemp[tile_x / 2 + 2 + k * f / 2].y * thetaTemp[tile_y / 2 + 4 + k * f / 2].y;\
        temp60 += thetaTemp[tile_x / 2 + 3 + k * f / 2].x * thetaTemp[tile_y / 2 + 0 + k * f / 2].x;\
        temp61 += thetaTemp[tile_x / 2 + 3 + k * f / 2].x * thetaTemp[tile_y / 2 + 0 + k * f / 2].y;\
        temp62 += thetaTemp[tile_x / 2 + 3 + k * f / 2].x * thetaTemp[tile_y / 2 + 1 + k * f / 2].x;\
        temp63 += thetaTemp[tile_x / 2 + 3 + k * f / 2].x * thetaTemp[tile_y / 2 + 1 + k * f / 2].y;\
        temp64 += thetaTemp[tile_x / 2 + 3 + k * f / 2].x * thetaTemp[tile_y / 2 + 2 + k * f / 2].x;\
        temp65 += thetaTemp[tile_x / 2 + 3 + k * f / 2].x * thetaTemp[tile_y / 2 + 2 + k * f / 2].y;\
        temp66 += thetaTemp[tile_x / 2 + 3 + k * f / 2].x * thetaTemp[tile_y / 2 + 3 + k * f / 2].x;\
        temp67 += thetaTemp[tile_x / 2 + 3 + k * f / 2].x * thetaTemp[tile_y / 2 + 3 + k * f / 2].y;\
        temp68 += thetaTemp[tile_x / 2 + 3 + k * f / 2].x * thetaTemp[tile_y / 2 + 4 + k * f / 2].x;\
        temp69 += thetaTemp[tile_x / 2 + 3 + k * f / 2].x * thetaTemp[tile_y / 2 + 4 + k * f / 2].y;\
        temp70 += thetaTemp[tile_x / 2 + 3 + k * f / 2].y * thetaTemp[tile_y / 2 + 0 + k * f / 2].x;\
        temp71 += thetaTemp[tile_x / 2 + 3 + k * f / 2].y * thetaTemp[tile_y / 2 + 0 + k * f / 2].y;\
        temp72 += thetaTemp[tile_x / 2 + 3 + k * f / 2].y * thetaTemp[tile_y / 2 + 1 + k * f / 2].x;\
        temp73 += thetaTemp[tile_x / 2 + 3 + k * f / 2].y * thetaTemp[tile_y / 2 + 1 + k * f / 2].y;\
        temp74 += thetaTemp[tile_x / 2 + 3 + k * f / 2].y * thetaTemp[tile_y / 2 + 2 + k * f / 2].x;\
        temp75 += thetaTemp[tile_x / 2 + 3 + k * f / 2].y * thetaTemp[tile_y / 2 + 2 + k * f / 2].y;\
        temp76 += thetaTemp[tile_x / 2 + 3 + k * f / 2].y * thetaTemp[tile_y / 2 + 3 + k * f / 2].x;\
        temp77 += thetaTemp[tile_x / 2 + 3 + k * f / 2].y * thetaTemp[tile_y / 2 + 3 + k * f / 2].y;\
        temp78 += thetaTemp[tile_x / 2 + 3 + k * f / 2].y * thetaTemp[tile_y / 2 + 4 + k * f / 2].x;\
        temp79 += thetaTemp[tile_x / 2 + 3 + k * f / 2].y * thetaTemp[tile_y / 2 + 4 + k * f / 2].y;\
        temp80 += thetaTemp[tile_x / 2 + 4 + k * f / 2].x * thetaTemp[tile_y / 2 + 0 + k * f / 2].x;\
        temp81 += thetaTemp[tile_x / 2 + 4 + k * f / 2].x * thetaTemp[tile_y / 2 + 0 + k * f / 2].y;\
        temp82 += thetaTemp[tile_x / 2 + 4 + k * f / 2].x * thetaTemp[tile_y / 2 + 1 + k * f / 2].x;\
        temp83 += thetaTemp[tile_x / 2 + 4 + k * f / 2].x * thetaTemp[tile_y / 2 + 1 + k * f / 2].y;\
        temp84 += thetaTemp[tile_x / 2 + 4 + k * f / 2].x * thetaTemp[tile_y / 2 + 2 + k * f / 2].x;\
        temp85 += thetaTemp[tile_x / 2 + 4 + k * f / 2].x * thetaTemp[tile_y / 2 + 2 + k * f / 2].y;\
        temp86 += thetaTemp[tile_x / 2 + 4 + k * f / 2].x * thetaTemp[tile_y / 2 + 3 + k * f / 2].x;\
        temp87 += thetaTemp[tile_x / 2 + 4 + k * f / 2].x * thetaTemp[tile_y / 2 + 3 + k * f / 2].y;\
        temp88 += thetaTemp[tile_x / 2 + 4 + k * f / 2].x * thetaTemp[tile_y / 2 + 4 + k * f / 2].x;\
        temp89 += thetaTemp[tile_x / 2 + 4 + k * f / 2].x * thetaTemp[tile_y / 2 + 4 + k * f / 2].y;\
        temp90 += thetaTemp[tile_x / 2 + 4 + k * f / 2].y * thetaTemp[tile_y / 2 + 0 + k * f / 2].x;\
        temp91 += thetaTemp[tile_x / 2 + 4 + k * f / 2].y * thetaTemp[tile_y / 2 + 0 + k * f / 2].y;\
        temp92 += thetaTemp[tile_x / 2 + 4 + k * f / 2].y * thetaTemp[tile_y / 2 + 1 + k * f / 2].x;\
        temp93 += thetaTemp[tile_x / 2 + 4 + k * f / 2].y * thetaTemp[tile_y / 2 + 1 + k * f / 2].y;\
        temp94 += thetaTemp[tile_x / 2 + 4 + k * f / 2].y * thetaTemp[tile_y / 2 + 2 + k * f / 2].x;\
        temp95 += thetaTemp[tile_x / 2 + 4 + k * f / 2].y * thetaTemp[tile_y / 2 + 2 + k * f / 2].y;\
        temp96 += thetaTemp[tile_x / 2 + 4 + k * f / 2].y * thetaTemp[tile_y / 2 + 3 + k * f / 2].x;\
        temp97 += thetaTemp[tile_x / 2 + 4 + k * f / 2].y * thetaTemp[tile_y / 2 + 3 + k * f / 2].y;\
        temp98 += thetaTemp[tile_x / 2 + 4 + k * f / 2].y * thetaTemp[tile_y / 2 + 4 + k * f / 2].x;\
        temp99 += thetaTemp[tile_x / 2 + 4 + k * f / 2].y * thetaTemp[tile_y / 2 + 4 + k * f / 2].y;\
    }\
    while (0)

#define fill_upper_half_from_registers()\
    do\
    {\
        array[tile_offset + tile_x + 0 + (tile_y + 0) * f] = temp0;\
        array[tile_offset + tile_x + 0 + (tile_y + 1) * f] = temp1;\
        array[tile_offset + tile_x + 0 + (tile_y + 2) * f] = temp2;\
        array[tile_offset + tile_x + 0 + (tile_y + 3) * f] = temp3;\
        array[tile_offset + tile_x + 0 + (tile_y + 4) * f] = temp4;\
        array[tile_offset + tile_x + 0 + (tile_y + 5) * f] = temp5;\
        array[tile_offset + tile_x + 0 + (tile_y + 6) * f] = temp6;\
        array[tile_offset + tile_x + 0 + (tile_y + 7) * f] = temp7;\
        array[tile_offset + tile_x + 0 + (tile_y + 8) * f] = temp8;\
        array[tile_offset + tile_x + 0 + (tile_y + 9) * f] = temp9;\
        array[tile_offset + tile_x + 1 + (tile_y + 0) * f] = temp10;\
        array[tile_offset + tile_x + 1 + (tile_y + 1) * f] = temp11;\
        array[tile_offset + tile_x + 1 + (tile_y + 2) * f] = temp12;\
        array[tile_offset + tile_x + 1 + (tile_y + 3) * f] = temp13;\
        array[tile_offset + tile_x + 1 + (tile_y + 4) * f] = temp14;\
        array[tile_offset + tile_x + 1 + (tile_y + 5) * f] = temp15;\
        array[tile_offset + tile_x + 1 + (tile_y + 6) * f] = temp16;\
        array[tile_offset + tile_x + 1 + (tile_y + 7) * f] = temp17;\
        array[tile_offset + tile_x + 1 + (tile_y + 8) * f] = temp18;\
        array[tile_offset + tile_x + 1 + (tile_y + 9) * f] = temp19;\
        array[tile_offset + tile_x + 2 + (tile_y + 0) * f] = temp20;\
        array[tile_offset + tile_x + 2 + (tile_y + 1) * f] = temp21;\
        array[tile_offset + tile_x + 2 + (tile_y + 2) * f] = temp22;\
        array[tile_offset + tile_x + 2 + (tile_y + 3) * f] = temp23;\
        array[tile_offset + tile_x + 2 + (tile_y + 4) * f] = temp24;\
        array[tile_offset + tile_x + 2 + (tile_y + 5) * f] = temp25;\
        array[tile_offset + tile_x + 2 + (tile_y + 6) * f] = temp26;\
        array[tile_offset + tile_x + 2 + (tile_y + 7) * f] = temp27;\
        array[tile_offset + tile_x + 2 + (tile_y + 8) * f] = temp28;\
        array[tile_offset + tile_x + 2 + (tile_y + 9) * f] = temp29;\
        array[tile_offset + tile_x + 3 + (tile_y + 0) * f] = temp30;\
        array[tile_offset + tile_x + 3 + (tile_y + 1) * f] = temp31;\
        array[tile_offset + tile_x + 3 + (tile_y + 2) * f] = temp32;\
        array[tile_offset + tile_x + 3 + (tile_y + 3) * f] = temp33;\
        array[tile_offset + tile_x + 3 + (tile_y + 4) * f] = temp34;\
        array[tile_offset + tile_x + 3 + (tile_y + 5) * f] = temp35;\
        array[tile_offset + tile_x + 3 + (tile_y + 6) * f] = temp36;\
        array[tile_offset + tile_x + 3 + (tile_y + 7) * f] = temp37;\
        array[tile_offset + tile_x + 3 + (tile_y + 8) * f] = temp38;\
        array[tile_offset + tile_x + 3 + (tile_y + 9) * f] = temp39;\
        array[tile_offset + tile_x + 4 + (tile_y + 0) * f] = temp40;\
        array[tile_offset + tile_x + 4 + (tile_y + 1) * f] = temp41;\
        array[tile_offset + tile_x + 4 + (tile_y + 2) * f] = temp42;\
        array[tile_offset + tile_x + 4 + (tile_y + 3) * f] = temp43;\
        array[tile_offset + tile_x + 4 + (tile_y + 4) * f] = temp44;\
        array[tile_offset + tile_x + 4 + (tile_y + 5) * f] = temp45;\
        array[tile_offset + tile_x + 4 + (tile_y + 6) * f] = temp46;\
        array[tile_offset + tile_x + 4 + (tile_y + 7) * f] = temp47;\
        array[tile_offset + tile_x + 4 + (tile_y + 8) * f] = temp48;\
        array[tile_offset + tile_x + 4 + (tile_y + 9) * f] = temp49;\
        array[tile_offset + tile_x + 5 + (tile_y + 0) * f] = temp50;\
        array[tile_offset + tile_x + 5 + (tile_y + 1) * f] = temp51;\
        array[tile_offset + tile_x + 5 + (tile_y + 2) * f] = temp52;\
        array[tile_offset + tile_x + 5 + (tile_y + 3) * f] = temp53;\
        array[tile_offset + tile_x + 5 + (tile_y + 4) * f] = temp54;\
        array[tile_offset + tile_x + 5 + (tile_y + 5) * f] = temp55;\
        array[tile_offset + tile_x + 5 + (tile_y + 6) * f] = temp56;\
        array[tile_offset + tile_x + 5 + (tile_y + 7) * f] = temp57;\
        array[tile_offset + tile_x + 5 + (tile_y + 8) * f] = temp58;\
        array[tile_offset + tile_x + 5 + (tile_y + 9) * f] = temp59;\
        array[tile_offset + tile_x + 6 + (tile_y + 0) * f] = temp60;\
        array[tile_offset + tile_x + 6 + (tile_y + 1) * f] = temp61;\
        array[tile_offset + tile_x + 6 + (tile_y + 2) * f] = temp62;\
        array[tile_offset + tile_x + 6 + (tile_y + 3) * f] = temp63;\
        array[tile_offset + tile_x + 6 + (tile_y + 4) * f] = temp64;\
        array[tile_offset + tile_x + 6 + (tile_y + 5) * f] = temp65;\
        array[tile_offset + tile_x + 6 + (tile_y + 6) * f] = temp66;\
        array[tile_offset + tile_x + 6 + (tile_y + 7) * f] = temp67;\
        array[tile_offset + tile_x + 6 + (tile_y + 8) * f] = temp68;\
        array[tile_offset + tile_x + 6 + (tile_y + 9) * f] = temp69;\
        array[tile_offset + tile_x + 7 + (tile_y + 0) * f] = temp70;\
        array[tile_offset + tile_x + 7 + (tile_y + 1) * f] = temp71;\
        array[tile_offset + tile_x + 7 + (tile_y + 2) * f] = temp72;\
        array[tile_offset + tile_x + 7 + (tile_y + 3) * f] = temp73;\
        array[tile_offset + tile_x + 7 + (tile_y + 4) * f] = temp74;\
        array[tile_offset + tile_x + 7 + (tile_y + 5) * f] = temp75;\
        array[tile_offset + tile_x + 7 + (tile_y + 6) * f] = temp76;\
        array[tile_offset + tile_x + 7 + (tile_y + 7) * f] = temp77;\
        array[tile_offset + tile_x + 7 + (tile_y + 8) * f] = temp78;\
        array[tile_offset + tile_x + 7 + (tile_y + 9) * f] = temp79;\
        array[tile_offset + tile_x + 8 + (tile_y + 0) * f] = temp80;\
        array[tile_offset + tile_x + 8 + (tile_y + 1) * f] = temp81;\
        array[tile_offset + tile_x + 8 + (tile_y + 2) * f] = temp82;\
        array[tile_offset + tile_x + 8 + (tile_y + 3) * f] = temp83;\
        array[tile_offset + tile_x + 8 + (tile_y + 4) * f] = temp84;\
        array[tile_offset + tile_x + 8 + (tile_y + 5) * f] = temp85;\
        array[tile_offset + tile_x + 8 + (tile_y + 6) * f] = temp86;\
        array[tile_offset + tile_x + 8 + (tile_y + 7) * f] = temp87;\
        array[tile_offset + tile_x + 8 + (tile_y + 8) * f] = temp88;\
        array[tile_offset + tile_x + 8 + (tile_y + 9) * f] = temp89;\
        array[tile_offset + tile_x + 9 + (tile_y + 0) * f] = temp90;\
        array[tile_offset + tile_x + 9 + (tile_y + 1) * f] = temp91;\
        array[tile_offset + tile_x + 9 + (tile_y + 2) * f] = temp92;\
        array[tile_offset + tile_x + 9 + (tile_y + 3) * f] = temp93;\
        array[tile_offset + tile_x + 9 + (tile_y + 4) * f] = temp94;\
        array[tile_offset + tile_x + 9 + (tile_y + 5) * f] = temp95;\
        array[tile_offset + tile_x + 9 + (tile_y + 6) * f] = temp96;\
        array[tile_offset + tile_x + 9 + (tile_y + 7) * f] = temp97;\
        array[tile_offset + tile_x + 9 + (tile_y + 8) * f] = temp98;\
        array[tile_offset + tile_x + 9 + (tile_y + 9) * f] = temp99;\
    }\
    while (0)

#define fill_lower_half_from_registers()\
    do\
    {\
        array[tile_offset + tile_y + 0 + (tile_x + 0) * f] = temp0;\
        array[tile_offset + tile_y + 1 + (tile_x + 0) * f] = temp1;\
        array[tile_offset + tile_y + 2 + (tile_x + 0) * f] = temp2;\
        array[tile_offset + tile_y + 3 + (tile_x + 0) * f] = temp3;\
        array[tile_offset + tile_y + 4 + (tile_x + 0) * f] = temp4;\
        array[tile_offset + tile_y + 5 + (tile_x + 0) * f] = temp5;\
        array[tile_offset + tile_y + 6 + (tile_x + 0) * f] = temp6;\
        array[tile_offset + tile_y + 7 + (tile_x + 0) * f] = temp7;\
        array[tile_offset + tile_y + 8 + (tile_x + 0) * f] = temp8;\
        array[tile_offset + tile_y + 9 + (tile_x + 0) * f] = temp9;\
        array[tile_offset + tile_y + 0 + (tile_x + 1) * f] = temp10;\
        array[tile_offset + tile_y + 1 + (tile_x + 1) * f] = temp11;\
        array[tile_offset + tile_y + 2 + (tile_x + 1) * f] = temp12;\
        array[tile_offset + tile_y + 3 + (tile_x + 1) * f] = temp13;\
        array[tile_offset + tile_y + 4 + (tile_x + 1) * f] = temp14;\
        array[tile_offset + tile_y + 5 + (tile_x + 1) * f] = temp15;\
        array[tile_offset + tile_y + 6 + (tile_x + 1) * f] = temp16;\
        array[tile_offset + tile_y + 7 + (tile_x + 1) * f] = temp17;\
        array[tile_offset + tile_y + 8 + (tile_x + 1) * f] = temp18;\
        array[tile_offset + tile_y + 9 + (tile_x + 1) * f] = temp19;\
        array[tile_offset + tile_y + 0 + (tile_x + 2) * f] = temp20;\
        array[tile_offset + tile_y + 1 + (tile_x + 2) * f] = temp21;\
        array[tile_offset + tile_y + 2 + (tile_x + 2) * f] = temp22;\
        array[tile_offset + tile_y + 3 + (tile_x + 2) * f] = temp23;\
        array[tile_offset + tile_y + 4 + (tile_x + 2) * f] = temp24;\
        array[tile_offset + tile_y + 5 + (tile_x + 2) * f] = temp25;\
        array[tile_offset + tile_y + 6 + (tile_x + 2) * f] = temp26;\
        array[tile_offset + tile_y + 7 + (tile_x + 2) * f] = temp27;\
        array[tile_offset + tile_y + 8 + (tile_x + 2) * f] = temp28;\
        array[tile_offset + tile_y + 9 + (tile_x + 2) * f] = temp29;\
        array[tile_offset + tile_y + 0 + (tile_x + 3) * f] = temp30;\
        array[tile_offset + tile_y + 1 + (tile_x + 3) * f] = temp31;\
        array[tile_offset + tile_y + 2 + (tile_x + 3) * f] = temp32;\
        array[tile_offset + tile_y + 3 + (tile_x + 3) * f] = temp33;\
        array[tile_offset + tile_y + 4 + (tile_x + 3) * f] = temp34;\
        array[tile_offset + tile_y + 5 + (tile_x + 3) * f] = temp35;\
        array[tile_offset + tile_y + 6 + (tile_x + 3) * f] = temp36;\
        array[tile_offset + tile_y + 7 + (tile_x + 3) * f] = temp37;\
        array[tile_offset + tile_y + 8 + (tile_x + 3) * f] = temp38;\
        array[tile_offset + tile_y + 9 + (tile_x + 3) * f] = temp39;\
        array[tile_offset + tile_y + 0 + (tile_x + 4) * f] = temp40;\
        array[tile_offset + tile_y + 1 + (tile_x + 4) * f] = temp41;\
        array[tile_offset + tile_y + 2 + (tile_x + 4) * f] = temp42;\
        array[tile_offset + tile_y + 3 + (tile_x + 4) * f] = temp43;\
        array[tile_offset + tile_y + 4 + (tile_x + 4) * f] = temp44;\
        array[tile_offset + tile_y + 5 + (tile_x + 4) * f] = temp45;\
        array[tile_offset + tile_y + 6 + (tile_x + 4) * f] = temp46;\
        array[tile_offset + tile_y + 7 + (tile_x + 4) * f] = temp47;\
        array[tile_offset + tile_y + 8 + (tile_x + 4) * f] = temp48;\
        array[tile_offset + tile_y + 9 + (tile_x + 4) * f] = temp49;\
        array[tile_offset + tile_y + 0 + (tile_x + 5) * f] = temp50;\
        array[tile_offset + tile_y + 1 + (tile_x + 5) * f] = temp51;\
        array[tile_offset + tile_y + 2 + (tile_x + 5) * f] = temp52;\
        array[tile_offset + tile_y + 3 + (tile_x + 5) * f] = temp53;\
        array[tile_offset + tile_y + 4 + (tile_x + 5) * f] = temp54;\
        array[tile_offset + tile_y + 5 + (tile_x + 5) * f] = temp55;\
        array[tile_offset + tile_y + 6 + (tile_x + 5) * f] = temp56;\
        array[tile_offset + tile_y + 7 + (tile_x + 5) * f] = temp57;\
        array[tile_offset + tile_y + 8 + (tile_x + 5) * f] = temp58;\
        array[tile_offset + tile_y + 9 + (tile_x + 5) * f] = temp59;\
        array[tile_offset + tile_y + 0 + (tile_x + 6) * f] = temp60;\
        array[tile_offset + tile_y + 1 + (tile_x + 6) * f] = temp61;\
        array[tile_offset + tile_y + 2 + (tile_x + 6) * f] = temp62;\
        array[tile_offset + tile_y + 3 + (tile_x + 6) * f] = temp63;\
        array[tile_offset + tile_y + 4 + (tile_x + 6) * f] = temp64;\
        array[tile_offset + tile_y + 5 + (tile_x + 6) * f] = temp65;\
        array[tile_offset + tile_y + 6 + (tile_x + 6) * f] = temp66;\
        array[tile_offset + tile_y + 7 + (tile_x + 6) * f] = temp67;\
        array[tile_offset + tile_y + 8 + (tile_x + 6) * f] = temp68;\
        array[tile_offset + tile_y + 9 + (tile_x + 6) * f] = temp69;\
        array[tile_offset + tile_y + 0 + (tile_x + 7) * f] = temp70;\
        array[tile_offset + tile_y + 1 + (tile_x + 7) * f] = temp71;\
        array[tile_offset + tile_y + 2 + (tile_x + 7) * f] = temp72;\
        array[tile_offset + tile_y + 3 + (tile_x + 7) * f] = temp73;\
        array[tile_offset + tile_y + 4 + (tile_x + 7) * f] = temp74;\
        array[tile_offset + tile_y + 5 + (tile_x + 7) * f] = temp75;\
        array[tile_offset + tile_y + 6 + (tile_x + 7) * f] = temp76;\
        array[tile_offset + tile_y + 7 + (tile_x + 7) * f] = temp77;\
        array[tile_offset + tile_y + 8 + (tile_x + 7) * f] = temp78;\
        array[tile_offset + tile_y + 9 + (tile_x + 7) * f] = temp79;\
        array[tile_offset + tile_y + 0 + (tile_x + 8) * f] = temp80;\
        array[tile_offset + tile_y + 1 + (tile_x + 8) * f] = temp81;\
        array[tile_offset + tile_y + 2 + (tile_x + 8) * f] = temp82;\
        array[tile_offset + tile_y + 3 + (tile_x + 8) * f] = temp83;\
        array[tile_offset + tile_y + 4 + (tile_x + 8) * f] = temp84;\
        array[tile_offset + tile_y + 5 + (tile_x + 8) * f] = temp85;\
        array[tile_offset + tile_y + 6 + (tile_x + 8) * f] = temp86;\
        array[tile_offset + tile_y + 7 + (tile_x + 8) * f] = temp87;\
        array[tile_offset + tile_y + 8 + (tile_x + 8) * f] = temp88;\
        array[tile_offset + tile_y + 9 + (tile_x + 8) * f] = temp89;\
        array[tile_offset + tile_y + 0 + (tile_x + 9) * f] = temp90;\
        array[tile_offset + tile_y + 1 + (tile_x + 9) * f] = temp91;\
        array[tile_offset + tile_y + 2 + (tile_x + 9) * f] = temp92;\
        array[tile_offset + tile_y + 3 + (tile_x + 9) * f] = temp93;\
        array[tile_offset + tile_y + 4 + (tile_x + 9) * f] = temp94;\
        array[tile_offset + tile_y + 5 + (tile_x + 9) * f] = temp95;\
        array[tile_offset + tile_y + 6 + (tile_x + 9) * f] = temp96;\
        array[tile_offset + tile_y + 7 + (tile_x + 9) * f] = temp97;\
        array[tile_offset + tile_y + 8 + (tile_x + 9) * f] = temp98;\
        array[tile_offset + tile_y + 9 + (tile_x + 9) * f] = temp99;\
    }\
    while (0)

#endif /* REG_OPS_H_ */
