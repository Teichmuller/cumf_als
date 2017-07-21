/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * als.h
 *
 *  Created on: Aug 13, 2015
 *      Author: weitan
 */
// fp16 is permanently removed			----Xuan
#ifndef ALS_H_
#define ALS_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <host_defines.h>

//these parameters do not change among different problem size
//our kernels handle the case where F%T==0 and F = 100

#define USE_DOUBLE
#ifdef USE_DOUBLE
using dtype = double;
using dtype2 = double2;
#else
using dtype = float;
using dtype2 = float2;
#endif

#define SCAN_BATCH 28
#define SURPASS_NAN
#define MEASURE_PERF

#include "reg_ops.h"

#define cudacall(call) \
    do\
    {\
	cudaError_t err = (call);\
	if(cudaSuccess != err)\
	    {\
		fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	    }\
    }\
    while (0)\

#define cublascall(call) \
do\
{\
	cublasStatus_t status = (call);\
	if(CUBLAS_STATUS_SUCCESS != status)\
	{\
		fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	}\
}\
while(0)\

#define cusparsecall(call) \
do\
{\
	cusparseStatus_t status = (call);\
	if(CUSPARSE_STATUS_SUCCESS != status)\
	{\
		fprintf(stderr,"CUSPARSE Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	}\
}\
while(0)\

#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }\
while(0)\

dtype doALS(const int* csrRowIndexHostPtr, const int* csrColIndexHostPtr, const dtype* csrValHostPtr,
		const int* cscRowIndexHostPtr, const int* cscColIndexHostPtr, const dtype* cscValHostPtr,
		const int* cooRowIndexHostPtr, dtype* thetaTHost, dtype * XTHost,
		const int * cooRowIndexTestHostPtr, const int * cooColIndexTestHostPtr, const dtype * cooValHostTestPtr,
		const int m, const int n, const int f, const long nnz, const long nnz_test, const dtype lambda, const dtype mu,
		const int ITERS, const int X_BATCH, const int THETA_BATCH, const int DEVICEID, const bool verbose);

#endif /* ALS_H_ */
