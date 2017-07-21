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
 * als.cu
 *
 *  Created on: Feb 10, 2015
 *      Author: Wei Tan (wtan@us.ibm.com)
 *  Alternating Least Square for Matrix Factorization on CUDA 7.0+
 *  Code optimized for F = 100, and on cc 3.5, 3.7 platforms. Also tested in cc 5.2
 */
// fp16 is permanently removed			----Xuan
#include "als.h"
#include "device_utilities.h"
#include "host_utilities.h"
#include <fstream>
#include <assert.h>
#include <iostream>

//#define USE_CG
//#define CG_ITER 6
//#include "cg.h"


using namespace std;

__global__ void Smoothing(const int batch_offset, const int m, const int f, const dtype mu,
		dtype * ythetaT, dtype * XT)
{
	int row = batch_offset + blockIdx.x;
	int col = threadIdx.x;
	int pos = row * f + col;
	dtype addend = 0;
	if (row != 0) {
		int prev_pos = (row - 1) * f + col;
		addend += XT[prev_pos];
	}
	if (row != m - 1) {
		int next_pos = (row + 1) * f + col;
		addend += XT[next_pos];
	}
	ythetaT[pos] += mu * addend;
}

__global__ void Smoothing_rev1(const int batch_offset, const int m, const int f, const dtype mu,
		dtype * ythetaT, dtype * XT)
{
	extern __shared__ dtype tempXT[];

	int row = blockIdx.x + batch_offset;
	int col = threadIdx.x;

	int pos_start = row * f;

	int pos = pos_start + col;
	int prev_pos = pos - f;
	int next_pos = pos + f;

	int local_pos = col + f;
	int local_prev_pos = col;
	int local_next_pos = local_pos + f;
	// gmem -> smem
	tempXT[local_pos] = ythetaT[pos];
	if (row != 0)
		tempXT[local_prev_pos] = XT[prev_pos];
	if (row != m - 1)
		tempXT[local_next_pos] = XT[next_pos];
	__syncthreads();
	// calc
	dtype addend = 0;
	if (row != 0)
		addend += tempXT[local_prev_pos];
	if (row != m - 1)
		addend += tempXT[local_next_pos];
	tempXT[local_pos] += mu * addend;
	__syncthreads();
	// smem -> gmem
	ythetaT[pos] = tempXT[local_pos];

}

int updateX(const int batch_size, const int batch_offset, dtype * ythetaT, dtype * tt, dtype * XT,
		cublasHandle_t handle, const int m, const int n, const int f, const int nnz, const dtype mu,
		dtype** devPtrTTHost, dtype **devPtrYthetaTHost){
	#ifdef DEBUG
	dtype elapsed;
	struct timeval tv0, tv1, tv2;
	gettimeofday(&tv0, NULL);
	printf("*******Batch LU factorization of tt.\n");
	#endif
	//pointers needed by batch op
	dtype **devPtrTT = 0;
	int *INFO;
	for (int k = 0; k < batch_size; k++) {
		devPtrTTHost[k] = &tt[k * f * f];
	}
	cudacall(cudaMalloc((void** ) &devPtrTT, batch_size * sizeof(*devPtrTT)));
	cudacall(cudaMemcpy(devPtrTT, devPtrTTHost, batch_size * sizeof(*devPtrTT),cudaMemcpyHostToDevice));
	//cudacall( cudaMalloc(&P, f * batch_size * sizeof(int)) );
	cudacall( cudaMalloc(&INFO, batch_size * sizeof(int) ));
#ifdef USE_DOUBLE
	cublascall(cublasDgetrfBatched(handle, f, devPtrTT, f, NULL, INFO, batch_size));
#else
	cublascall(cublasSgetrfBatched(handle, f, devPtrTT, f, NULL, INFO, batch_size));
#endif

	cudaThreadSynchronize();
	#ifdef DEBUG
	gettimeofday(&tv1, NULL);
	elapsed = (tv1.tv_sec - tv0.tv_sec)
			+ (tv1.tv_usec - tv0.tv_usec) / 1000000.0;
	printf("\t %f seconds. \n", elapsed);

	printf("*******solve: tt * XT = ythetaT use cublas, with LU decomposition.\n");
	#endif

	dtype **devPtrYthetaT = 0;

	Smoothing<<<batch_size, f>>>(batch_offset, m, f, mu, ythetaT, XT);
	//Smoothing_rev1<<<batch_size, f, 3 * f * sizeof(dtype)>>>(batch_offset, m, f, mu, ythetaT, XT);
	cudaThreadSynchronize();

	for (int k = 0; k < batch_size; k++) {
		devPtrYthetaTHost[k] = &ythetaT[batch_offset * f + k * f];
	}
	cudacall(cudaMalloc((void** ) &devPtrYthetaT, batch_size * sizeof(*devPtrYthetaT)));
	cudacall(cudaMemcpy(devPtrYthetaT, devPtrYthetaTHost, batch_size * sizeof(*devPtrYthetaT), cudaMemcpyHostToDevice));

	int * info2 = (int *) malloc(sizeof(int));
#ifdef USE_DOUBLE
	cublascall( cublasDgetrsBatched(handle, CUBLAS_OP_N, f, 1,
			(const dtype ** ) devPtrTT, f, NULL, devPtrYthetaT, f, info2, batch_size) );
#else
	cublascall( cublasSgetrsBatched(handle, CUBLAS_OP_N, f, 1,
			(const dtype ** ) devPtrTT, f, NULL, devPtrYthetaT, f, info2, batch_size) );
#endif

	cudaThreadSynchronize();
	cudaError_t cudaStat1 = cudaGetLastError();
	if (cudaStat1 != cudaSuccess) {
		fprintf(stderr,"Failed to launch cublasSgetrsBatched (error code: %s)!\n", cudaGetErrorString(cudaStat1));
		exit(EXIT_FAILURE);
	}

	cudacall( cudaMemcpy(&XT[batch_offset * f], &ythetaT[batch_offset * f],
			batch_size * f * sizeof(dtype), cudaMemcpyDeviceToDevice) );
	#ifdef DEBUG
	gettimeofday(&tv2, NULL);
	elapsed = (tv2.tv_sec - tv1.tv_sec)
			+ (tv2.tv_usec - tv1.tv_usec) / 1000000.0;
	printf("\t %f seconds. \n", elapsed);
	#endif

	cudacall(cudaFree(devPtrTT));
	//cudacall(cudaFree(P));
	cudacall(cudaFree(INFO));
	cudacall(cudaFree(devPtrYthetaT));
	return 0;
}

int updateTheta(const int batch_size, const int batch_offset, dtype * yTXT,
		  dtype * xx, dtype * thetaT,
		cublasHandle_t handle,
		 const int m, const int n, const int f, const int nnz,
		 dtype ** devPtrXXHost, dtype **devPtrYTXTHost ){

	#ifdef DEBUG
	dtype elapsed;
	struct timeval tv0, tv1, tv2;
	gettimeofday(&tv0, NULL);
	printf("*******LU factorize xx.\n");
	#endif
	dtype **devPtrXX = 0;

	for (int k = 0; k < batch_size; k++) {
		devPtrXXHost[k] = &xx[k * f * f];
	}
	cudacall(cudaMalloc((void** ) &devPtrXX, batch_size * sizeof(*devPtrXX)));
	cudacall(cudaMemcpy(devPtrXX, devPtrXXHost, batch_size * sizeof(*devPtrXX), cudaMemcpyHostToDevice));
	int *INFO;
	//cudacall(cudaMalloc(&P, f * batch_size * sizeof(int)));
	cudacall(cudaMalloc(&INFO, batch_size * sizeof(int)));
#ifdef USE_DOUBLE
	cublascall(cublasDgetrfBatched(handle, f, devPtrXX, f, NULL, INFO, batch_size));
#else
	cublascall(cublasSgetrfBatched(handle, f, devPtrXX, f, NULL, INFO, batch_size));
#endif
	cudaThreadSynchronize();
	#ifdef DEBUG
	gettimeofday(&tv1, NULL);
	elapsed = (tv1.tv_sec - tv0.tv_sec)
			+ (tv1.tv_usec - tv0.tv_usec) / 1000000.0;
	printf("\t %f seconds. \n", elapsed);

	printf("******* solve xx * thetaT = yTXT with CUDA 7.\n");
	#endif
	dtype **devPtrYTXT = 0;

	for (int k = 0; k < batch_size; k++) {
		devPtrYTXTHost[k] = &yTXT[batch_offset * f + k * f];
	}

	cudacall(cudaMalloc((void** ) &devPtrYTXT, batch_size * sizeof(*devPtrYTXT)));
	cudacall(cudaMemcpy(devPtrYTXT, devPtrYTXTHost, batch_size * sizeof(*devPtrYTXT),cudaMemcpyHostToDevice));

	int * info2 = (int *) malloc(sizeof(int));
#ifdef USE_DOUBLE
	cublascall( cublasDgetrsBatched(handle, CUBLAS_OP_N, f, 1,
			(const dtype ** ) devPtrXX, f, NULL, devPtrYTXT, f, info2, batch_size) );
#else
	cublascall( cublasSgetrsBatched(handle, CUBLAS_OP_N, f, 1,
			(const dtype ** ) devPtrXX, f, NULL, devPtrYTXT, f, info2, batch_size) );
#endif
	cudaThreadSynchronize();
	cudaError_t cudaStat1 = cudaGetLastError();
	if (cudaStat1 != cudaSuccess) {
		fprintf(stderr,"Failed to launch cublasSgetrsBatched (error code: %s)!\n", cudaGetErrorString(cudaStat1));
		exit(EXIT_FAILURE);
	}

	cudacall( cudaMemcpy( &thetaT[batch_offset * f], &yTXT[batch_offset * f],
	                        batch_size * f * sizeof(dtype), cudaMemcpyDeviceToDevice) );
	#ifdef DEBUG
	gettimeofday(&tv2, NULL);
	elapsed = (tv2.tv_sec - tv1.tv_sec)
			+ (tv2.tv_usec - tv1.tv_usec) / 1000000.0;
	printf("\t %f seconds. \n", elapsed);
	#endif

	cudaFree(devPtrXX);
	cudaFree(INFO);
	free(info2);
	cudaFree(devPtrYTXT);
	return 0;
}

__global__ void RMSE(const dtype * csrVal, const int* cooRowIndex,
		const int* csrColIndex, const dtype * __restrict__ thetaT, const dtype * __restrict__ XT, dtype * error, const int nnz,
		const int error_size, const int f) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < nnz) {
		int row = cooRowIndex[i];
		int col = csrColIndex[i];
		dtype e = csrVal[i];
		//if(i%1000000==0) printf("row: %d, col: %d, csrVal[%d]: %f.\n", row, col, i, e);
		for (int k = 0; k < f; k++) {
			#ifdef SURPASS_NAN
			//a and b could be; there are user/item in testing but not training set
			dtype a = __ldg(&thetaT[f * col + k]);
			dtype b = __ldg(&XT[f * row + k]);
			//if(isnan(a)||isnan(b))//nan not working in some platform
			if(a!=a||b!=b)
				break;
			else
				e -= a * b;
			//if(isnan(a)) printf("row: %d, col: %d\n", row, col);
			//if(isnan(b)) printf("b[%d]: %f.\n", i, b);
			#else
			e -= __ldg(&thetaT[f * col + k]) * __ldg(&XT[f * row + k]);
			#endif
		}
#ifdef USE_DOUBLE
		atomicDAdd(&error[i%error_size], e*e);
#else
		atomicAdd(&error[i%error_size], e*e);
#endif
		//if(i%1000000==0) printf("error[%d]: %f.\n", i, e);
	}
}

__global__ void MAE(const dtype * csrVal, const int* cooRowIndex,
		const int* csrColIndex, const dtype * __restrict__ thetaT, const dtype * __restrict__ XT, dtype * error, const int nnz,
		const int error_size, const int f)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < nnz)
	{
		int row = cooRowIndex[i];
		int col = csrColIndex[i];
		dtype e = csrVal[i];
		for (int k = 0; k < f; k++)
		{
			#ifdef SURPASS_NAN
				//a and b could be; there are user/item in testing but not training set
				dtype a = __ldg(&thetaT[f * col + k]);
				dtype b = __ldg(&XT[f * row + k]);
				//if(isnan(a)||isnan(b))//nan not working in some platform
				if (a != a || b != b)
				{
					e = 0;
					break;
				}
				else
					e -= a * b;
			#else
				e -= __ldg(&thetaT[f * col + k]) * __ldg(&XT[f * row + k]);
			#endif
		}
#ifdef USE_DOUBLE
		atomicDAdd(&error[i%error_size], fabs(e)); // overloading atomicAdd(double*, double) causes conflict
#else
		atomicAdd(&error[i%error_size], fabsf(e));
#endif
	}
}
/*
__global__ void
__launch_bounds__(64)
get_hermitian100(const int batch_offset, dtype2* tt,
		const int* csrRowIndex, const int* csrColIndex, const dtype lambda, const int m, const int f,
		const dtype2* __restrict__ thetaT) {
	extern __shared__ dtype2 thetaTemp[];
	int row = blockIdx.x + batch_offset;
	if (row < m) {
		//this block needs to handle end - start thetaT columns
		int start = csrRowIndex[row];
		int end = csrRowIndex[row + 1];
		//slide through [start, end] by window size SCAN_BATCH
		int iterations = (end - start - 1) / SCAN_BATCH + 1;
		dtype temp0= 0, temp1= 0, temp2= 0, temp3= 0, temp4= 0, temp5= 0, temp6= 0, temp7= 0, temp8= 0, temp9 = 0;
		dtype temp10= 0, temp11= 0, temp12= 0, temp13= 0, temp14= 0, temp15= 0, temp16= 0, temp17= 0, temp18= 0, temp19 = 0;
		dtype temp20= 0, temp21= 0, temp22= 0, temp23= 0, temp24= 0, temp25= 0, temp26= 0, temp27= 0, temp28= 0, temp29 = 0;
		dtype temp30= 0, temp31= 0, temp32= 0, temp33= 0, temp34= 0, temp35= 0, temp36= 0, temp37= 0, temp38= 0, temp39 = 0;
		dtype temp40= 0, temp41= 0, temp42= 0, temp43= 0, temp44= 0, temp45= 0, temp46= 0, temp47= 0, temp48= 0, temp49 = 0;
		dtype temp50= 0, temp51= 0, temp52= 0, temp53= 0, temp54= 0, temp55= 0, temp56= 0, temp57= 0, temp58= 0, temp59 = 0;
		dtype temp60= 0, temp61= 0, temp62= 0, temp63= 0, temp64= 0, temp65= 0, temp66= 0, temp67= 0, temp68= 0, temp69 = 0;
		dtype temp70= 0, temp71= 0, temp72= 0, temp73= 0, temp74= 0, temp75= 0, temp76= 0, temp77= 0, temp78= 0, temp79 = 0;
		dtype temp80= 0, temp81= 0, temp82= 0, temp83= 0, temp84= 0, temp85= 0, temp86= 0, temp87= 0, temp88= 0, temp89 = 0;
		dtype temp90= 0, temp91= 0, temp92= 0, temp93= 0, temp94= 0, temp95= 0, temp96= 0, temp97= 0, temp98= 0, temp99 = 0;

		int tile_x = 0;
		int tile_y = 0;

		int tile = f/10;
		for ( int i = 0; i < 10; i++){
			int end = ((20-i)*(i+1))/2;
			if(threadIdx.x < end){
				tile_x = i * tile;
				tile_y = (10 + threadIdx.x - end) * tile;
				break;
			}
		}
		//iteration: copy gmem-->smem; aggregate smem-->register
		for (int iter = 0; iter < iterations; iter ++){
			//copy texture --> smem, and sync
			/*
			This is the fastest implementation
			thetaT is NOT coalesced loaded but cached by L1 and L2
			faster than coalesced version (see the next paragraph commented out) 
			because it concurrently load multiple thetaT columns
			two threads per theta column, e.g., threads 0 & 1 for theta[0], threads 2 & 3 for theta[1]
			require: blockDim.x (64) >= 2*SCAN_BATCH
			*/
/*
			if(threadIdx.x < 2*SCAN_BATCH){
				int anchor = start + iter*SCAN_BATCH + threadIdx.x/2;
				if(anchor < end){
					int col = csrColIndex[anchor];
					//IMPORTANT: for loop has constant and identical start and end
					for (int k = 0; k < 50; k += 2)
						//thetaTemp[threadIdx.x*F/4 + k/2] =__ldg(&thetaT[ F/2 * col + threadIdx.x%2*F/4 + k/2]);
						thetaTemp[threadIdx.x*f/4 + k/2] = thetaT[ f/2 * col + threadIdx.x%2*f/4 + k/2];
				}
			}
*/

/*			
				//coalesced load thetaT, has to load column by column, less concurrency, worse performance
				int anchor = start + iter*SCAN_BATCH + threadIdx.x%32;
				int col_local;
				if(anchor < end && threadIdx.x%32 < SCAN_BATCH)
					col_local = csrColIndex[anchor];
				int stop = (end - start - iter*SCAN_BATCH < SCAN_BATCH)? end - start - iter*SCAN_BATCH: SCAN_BATCH;
				for (int k = 0; k < stop; k++){
					//deal with col_local in lane[k]
					int col = __shfl(col_local, k);
					//if(blockIdx.x==0 && threadIdx.x==0)
					//	printf("iter=%d,k=%d,col=%d,stop=%d,anchor=%d\n", iter,k, col, stop, anchor);
					//this type of for is bad in performance
					//for(int i = threadIdx.x; i < F; i += 64)
					if(threadIdx.x<F/2)
						thetaTemp[k*F/2 + threadIdx.x] = __ldg(&thetaT[ F/2 * col + threadIdx.x]);
				}
*/
/*
			__syncthreads();
			//tile: 10*10
			if(threadIdx.x < 55){
				if(iter < iterations - 1){
					for(int k = 0; k < SCAN_BATCH; k++)
						accumulate_in_registers();
				}
				else{
					for(int k = 0; k < end - start - iter*SCAN_BATCH; k++)
						accumulate_in_registers();
				}
				
			}
*//*
		}
		//end of iteration in copying from smem and aggregating in register
		__syncthreads();
		#ifdef DEBUG
		//if(threadIdx.x==0)
		//	printf("***temp 0~9: %f %f %f %f %f %f %f %f %f %f\n", temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9);
		#endif
		if(threadIdx.x < 55 ){
			//weighted-lambda regularization
			if(tile_x == tile_y){
				dtype temp = (end - start) * lambda;
				temp0 += temp;
				temp11 += temp;
				temp22 += temp;
				temp33 += temp;
				temp44 += temp;
				temp55 += temp;
				temp66 += temp;
				temp77 += temp;
				temp88 += temp;
				temp99 += temp;
			}
			//copy output to gmem
			int index = blockIdx.x*f*f/2;
			//fill_lower_half_from_registers();
#ifdef USE_DOUBLE
			fill_lower_half_from_registers_double2();
#else
			fill_lower_half_from_registers_float2();
#endif
			//symmetric
			if(tile_x!=tile_y){
				//fill_upper_half_from_registers();
#ifdef USE_DOUBLE
				fill_upper_half_from_registers_double2();
#else
				fill_upper_half_from_registers_float2();
#endif
			}
		}
	}
}*/

/*a generic kernel to get the hermitian matrices
 * as the left-hand side of the equations, to update X in ALS
 *examplary F = 100, T = 10
 */
__global__ void get_hermitianT10(const int batch_offset_row, dtype* array,
		const int* d_csr_RI, const int* d_csr_CI, const dtype lambda, const dtype mu, const int m, const int f,
		const dtype* __restrict__ d_thetaT)
{
	extern __shared__ dtype2 thetaTemp[];
	int row = blockIdx.x + batch_offset_row;
	if (row < m)
	{
		//this block needs to handle end - start thetaT columns
		int start = d_csr_RI[row];
		int end = d_csr_RI[row + 1];
		//slide through [start, end] by window size SCAN_BATCH
		int iterations = (end - start - 1) / SCAN_BATCH + 1;
		declare_registers();

		int N = f / RegisterTileSize; // N = 100 / 10=10; for F = 100 and T = 10
		int F = f; // forward
		int effective_block_size = N * (N + 1) / 2;

		//get the coordinates of this tile
		int tile_x = 0; //row
		int tile_y = 0;
		for (int i = 0; i < N; i++)
		{
			int end = ((2 * N - i) * (i + 1)) / 2;
			if (threadIdx.x < end)
			{
				tile_x = i * RegisterTileSize;
				tile_y = (N + threadIdx.x - end) * RegisterTileSize;
				break;
			}
		}
		int tile_offset = blockIdx.x * f * f; // offset of tile in the whole array
		int index = tile_offset; // forward

		//iteration: copy gmem-->smem; aggregate smem-->register
		for (int iter = 0; iter < iterations; iter++)
		{
			//phase 1 in iteration: gmem --> smem
			//REQ: blockDim.x >= F/2 (to load all columns of theta)
			if (threadIdx.x < f/2)
			{
				for (int k = 0; k< SCAN_BATCH; k++)
				{
					if (iter * SCAN_BATCH + k < end - start)
					{
						dtype2 theta;
						theta.x = __ldg(&d_thetaT[f * d_csr_CI[start + iter * SCAN_BATCH + k] + 2 * threadIdx.x]);
						theta.y = __ldg(&d_thetaT[f * d_csr_CI[start + iter * SCAN_BATCH + k] + 2 * threadIdx.x + 1]);
						thetaTemp[k * f / 2 + threadIdx.x] = theta;
					}
					else
						memset(&thetaTemp[k * f / 2 + threadIdx.x], 0, 2 * sizeof(dtype)); // not enough theta to copy, set zero
				}
			}			
			__syncthreads();
			
			//phase 2 in iteration: smem --> register
			if(threadIdx.x < effective_block_size) //this redundant "if" seems improving kernel performance
			{
				for(int k = 0; k < SCAN_BATCH; k++)
				{
					accumulate_in_registers();
				}
			}
		}
		//end of iteration in copying from smem and aggregating in register
		__syncthreads();

		//phase 3, after iteration: register --> gmem
		if(threadIdx.x < effective_block_size)
		{
			fill_lower_half_from_registers();

			//symmetric
			if(tile_x != tile_y)
			{
				fill_upper_half_from_registers();
			}

			//regularization
			if(tile_x == tile_y)
			{
				dtype addend = lambda + (row != 0 && row != m - 1 ? 2 * mu : mu);
				for(int k = 0; k < RegisterTileSize; k++)
					if (row != 0 && row != m - 1)
						array[index + (tile_x + k) * (1 + F)] += addend;
					else
						array[index + (tile_x + k) * (1 + F)] += addend;
			}
		}
	}
}


dtype doALS(const int *h_csr_RI, const int *h_csr_CI, const dtype *h_csr_Val,
		const int *h_csc_RI, const int *h_csc_CI, const dtype *h_csc_Val,
		const int *h_coo_RI, dtype *h_thetaT, dtype *h_xT,
		const int *h_test_coo_RI, const int *h_test_coo_CI, const dtype *h_test_coo_Val,
		const int m, const int n, const int f, const long nnz, const long nnz_test,
		const dtype lambda, const dtype mu,
		const int iters, const int n_x_batch, const int n_theta_batch, const int DEVICE_ID, const bool verbose = false)
{
	cudaSetDevice(DEVICE_ID);
	////////////////////////////////////////////////////////////////////////////////////////////////////
	if (verbose) printf("****** Training Parameters: m: %d, n:  %d, f: %d, lambda: %f, mu: %f, nnz: %ld ******\n", m, n, f, lambda, mu, nnz);
	if (verbose) printf("****** Testing Parameters: nnz_test: %ld ******\n", nnz_test);
	//device pointers
	int *d_csr_RI = 0;
	int *d_csr_CI = 0;
	int *d_csc_RI = 0;
	int *d_csc_CI = 0;
	dtype *d_csr_Val = 0;
	dtype *d_csc_Val = 0;
	dtype *d_thetaT = 0;
	dtype *d_xT = 0;
	dtype *d_a_array = 0;
	//coo to calculate error
	int *d_coo_RI = 0;
	int *d_test_coo_RI;
	int *d_test_coo_CI;
	dtype *d_test_coo_Val;
	dtype ret_error_test = 0;

	if (verbose) printf("****** Allocating GPU Memory...\n");
	cudacall(cudaMalloc((void**)&d_csc_RI, nnz * sizeof(d_csc_RI[0])));
	cudacall(cudaMalloc((void**)&d_csc_CI, (n + 1) * sizeof(d_csc_CI[0])));
	cudacall(cudaMalloc((void**)&d_csc_Val, nnz * sizeof(d_csc_Val[0])));
	cudacall(cudaMalloc((void**)&d_thetaT, f * n * sizeof(d_thetaT[0]))); // dim: f * n
	cudacall(cudaMalloc((void**)&d_xT, f * m * sizeof(d_xT[0]))); // dim: m * f

	if (verbose) printf("****** Copying to GPU Memory...\n");
	cudacall(cudaMemcpy(d_csc_RI, h_csc_RI, (size_t)nnz * sizeof(d_csc_RI[0]), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(d_csc_CI, h_csc_CI, (size_t)(n + 1) * sizeof(d_csc_CI[0]), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(d_csc_Val, h_csc_Val,(size_t)(nnz * sizeof(d_csc_Val[0])), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(d_thetaT, h_thetaT, (size_t)(n * f * sizeof(d_thetaT[0])), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(d_xT, h_xT, (size_t)(m * f * sizeof(d_xT[0])), cudaMemcpyHostToDevice)); // update x first, this line is not necessary

	if (verbose) printf("****** Configuring Device...\n");
	cudacall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	//64-bit smem access
	//http://acceleware.com/blog/maximizing-shared-memory-bandwidth-nvidia-kepler-gpus
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	if (verbose) printf("****** Initializing cuBLAS and cuSPARSE...\n");
	cublasHandle_t cublas_handle = 0;
	cublascall(cublasCreate(&cublas_handle));

	cusparseHandle_t cusparse_handle = 0;
	cusparseMatDescr_t cusparse_descr;
	cusparsecall(cusparseCreate(&cusparse_handle));
	cusparsecall( cusparseCreateMatDescr(&cusparse_descr));
	cusparseSetMatType(cusparse_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(cusparse_descr, CUSPARSE_INDEX_BASE_ZERO);

	#ifdef MEASURE_PERF
		// variable used to time
		double t0 = 0;
		double t1 = 0;
	#endif

	if (verbose) printf("****** Start Iterations...\n");
	for(int iter = 0; iter < iters; iter++)
	{
		// copy csr to device
		// TODO: Can this be outside of loop?
		cudacall(cudaMalloc((void**)&d_csr_RI, (m + 1) * sizeof(d_csr_RI[0])));
		cudacall(cudaMalloc((void**)&d_csr_CI, nnz * sizeof(d_csr_CI[0])));
		cudacall(cudaMalloc((void**)&d_csr_Val, nnz * sizeof(d_csr_Val[0])));
		cudacall(cudaMemcpy(d_csr_RI, h_csr_RI, (size_t)((m + 1) * sizeof(d_csr_RI[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(d_csr_CI, h_csr_CI, (size_t)(nnz * sizeof(d_csr_CI[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(d_csr_Val, h_csr_Val, (size_t)(nnz * sizeof(d_csr_Val[0])),cudaMemcpyHostToDevice));

		int dim_a_tile = f / RegisterTileSize * (f / RegisterTileSize + 1) / 2; // original: block_dim
		if (dim_a_tile < f / 2) dim_a_tile = f / 2;

		#ifdef MEASURE_PERF
			printf("------------------------------ iter %d / %d, update X ------------------------------\n", iter + 1, iters);
			t0 = seconds();
		#endif

		// ytheta = data * theta (dim: m * f)
		// however ythetaT (= (data * theta) ^ T) (dim: f * m) is the one needed
		// TODO: Can this be out side of loop?
		// TODO: can a single BLAS call do both multiplication and transposition?
		dtype *d_ytheta = 0;
		dtype *d_ythetaT = 0;
		cudacall(cudaMalloc((void**)&d_ytheta, f * m * sizeof(d_ytheta[0])));
		cudacall(cudaMalloc((void**)&d_ythetaT, f * m * sizeof(d_ythetaT[0])));

		const dtype alpha = 1.0f;
		const dtype beta = 0.0f;

		if (verbose) printf("****** Generating ythetaT...\n");
		#ifdef MEASURE_PERF
			t1 = seconds();
		#endif
#ifdef USE_DOUBLE
		cusparsecall(cusparseDcsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				CUSPARSE_OPERATION_TRANSPOSE, m, f, n, nnz, &alpha, cusparse_descr, d_csr_Val,
				d_csr_RI, d_csr_CI, d_thetaT, f, &beta, d_ytheta, m));
		cublascall(cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, f, m, &alpha,
				(const dtype*)d_ytheta, m, &beta, d_ythetaT, f, d_ythetaT, f));
#else
		cusparsecall(cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				CUSPARSE_OPERATION_TRANSPOSE, m, f, n, nnz, &alpha, cusparse_descr, d_csr_Val,
				d_csr_RI, d_csr_CI, d_thetaT, f, &beta, d_ytheta, m));
		cublascall(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, f, m, &alpha,
				(const dtype*)d_ytheta, m, &beta, d_ythetaT, f, d_ythetaT, f));
#endif
		#ifdef MEASURE_PERF
			printf("****** Generating ythetaT runs %f seconds.\n", seconds() - t1);
		#endif

		// TODO: ???
		cudacall(cudaFree(d_ytheta));
		cudacall(cudaFree(d_csr_Val));


		if (verbose) printf("****** Updating x...\n");
		for(int batch_id = 0; batch_id < n_x_batch; batch_id++)
		{
			#ifdef MEASURE_PERF
				printf("****** Batch %d / %d ******\n", batch_id + 1, n_x_batch);
			#endif

			int n_x_batch_row = 0; // original: batch_size
			if(n_x_batch_row != n_x_batch - 1)
				n_x_batch_row = m / n_x_batch;
			else
				n_x_batch_row = m - batch_id * (m / n_x_batch);
			int n_x_batch_offset_row = batch_id * (m / n_x_batch); // original: batch_offset

			cudacall(cudaMalloc((void**)&d_a_array, f * f * n_x_batch_row * sizeof(dtype)));

			#ifdef MEASURE_PERF
				printf("****** Get Hermitian x (batch %d / %d)\n", batch_id + 1, n_x_batch);
				t1 = seconds();
			#endif

			// I haven't added smoothing term to get_hermitian100		----Xuan
			//if(f == 100){
			//	get_hermitian100<<<batch_size, 64, SCAN_BATCH * f/2*sizeof(dtype2)>>>
			//		(batch_offset, (dtype2*)tt, csrRowIndex, csrColIndex, lambda, m, f, (dtype2*)thetaT);
			//}
			//else
				get_hermitianT10<<<n_x_batch_row, dim_a_tile, SCAN_BATCH * f / 2 * sizeof(dtype2)>>>
					(n_x_batch_offset_row, d_a_array, d_csr_RI, d_csr_CI, lambda, mu, m, f, d_thetaT);
			cudaDeviceSynchronize();
			cudaCheckError();
			#ifdef MEASURE_PERF
				printf("****** Get Hermitian kernel (batch %d / %d) runs %f seconds.\n", batch_id + 1, n_x_batch, seconds() - t1);
			#endif

			#ifdef MEASURE_PERF
				printf("****** Update x (batch %d / %d)\n", batch_id + 1, n_x_batch);
				t1 = seconds();
			#endif
			#ifdef USE_CG	// use CG iterative solver
				updateXWithCGHost(tt, &XT[batch_offset*f], &ythetaT[batch_offset*f], batch_size, f, CG_ITER);
			#else//use LU solver instead
				dtype **d_a_p_array = 0; // host pointers for cublas batch operations, original: devPtrTTHost
				dtype **d_ythetaT_p_array = 0; // original: devPtrYthetaTHost
				cudacall(cudaMallocHost((void**)&d_a_p_array, n_x_batch_row * sizeof(*d_a_p_array)));
				cudacall(cudaMallocHost((void**)&d_ythetaT_p_array, n_x_batch_row * sizeof(*d_ythetaT_p_array)));
				updateX(n_x_batch_row, n_x_batch_offset_row, d_ythetaT, d_a_array, d_xT, cublas_handle, m, n, f, nnz, mu, d_a_p_array, d_ythetaT_p_array);
				cudacall(cudaFreeHost(d_a_p_array));
				cudacall(cudaFreeHost(d_ythetaT_p_array));
			#endif
			#ifdef MEASURE_PERF
				printf("****** Update x solver (batch %d / %d) runs seconds: %f \n", batch_id + 1, n_x_batch, seconds() - t1);
			#endif
			cudacall(cudaFree(d_a_array));
		}
		#ifdef MEASURE_PERF
			printf("****** Update x runs %f seconds, gridSize: %d, blockSize %d.\n", seconds() - t0, m, f);
		#endif
		cudacall(cudaFree(d_csr_RI));
		cudacall(cudaFree(d_csr_CI));
		cudacall(cudaFree(d_ythetaT));

		#ifdef MEASURE_PERF
			printf("------------------------------ iter %d / %d, update theta ------------------------------\n", iter + 1, iters);
			t0 = seconds();
		#endif
		// yTx = (data ^ T) * x (dim: n * f)
		// however yTxT (= ((data ^ T) * x) ^ T, dim: f * n) is the one needed
		dtype *d_yTx = 0; // dim: n * f
		dtype *d_yTxT = 0; //
		cudacall(cudaMalloc((void**)&d_yTx, f * n * sizeof(d_yTx[0])));
		cudacall(cudaMalloc((void**)&d_yTxT, n * f * sizeof(d_yTxT[0])));

		#ifdef MEASURE_PERF
			printf("****** Generating yTxT...\n");
			t1 = seconds();
		#endif
#ifdef USE_DOUBLE
		cusparsecall( cusparseDcsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				CUSPARSE_OPERATION_TRANSPOSE, n, f, m, nnz, &alpha, cusparse_descr, d_csc_Val,
				d_csc_CI, d_csc_RI, d_xT, f, &beta, d_yTx, n));
		cublascall(cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, f, n, &alpha,
				(const dtype*)d_yTx, n, &beta, d_yTxT, f, d_yTxT, f));
#else
		cusparsecall( cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				CUSPARSE_OPERATION_TRANSPOSE, n, f, m, nnz, &alpha, cusparse_descr, d_csc_Val,
				d_csc_CI, d_csc_RI, d_xT, f, &beta, d_yTx, n));
		cublascall(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, f, n, &alpha,
				(const dtype*)d_yTx, n, &beta, d_yTxT, f, d_yTxT, f));
#endif
		#ifdef MEASURE_PERF
			printf("****** Generating yTxT runs %f seconds.\n", seconds() - t1);
		#endif

		cudaDeviceSynchronize();
		cudacall(cudaFree(d_yTx));

		if (verbose) printf("****** Updating theta...\n");
		for(int batch_id = 0; batch_id < n_theta_batch; batch_id++)
		{
			#ifdef MEASURE_PERF
				printf("****** Batch %d / %d.*******\n", batch_id + 1, n_theta_batch);
			#endif

			int n_theta_batch_row = 0; // original: batch_size
			if(batch_id != n_theta_batch - 1)
				n_theta_batch_row = n / n_theta_batch;
			else
				n_theta_batch_row = n - batch_id * (n / n_theta_batch);
			int n_theta_batch_offset_row = batch_id * (n / n_theta_batch); // original: batch_offset

			dtype *d_b_array = 0; //original: xx
			cudacall(cudaMalloc((void**)&d_b_array, f * f * n_theta_batch_row * sizeof(d_b_array[0])));
			cudacall(cudaMemset(d_b_array, 0, f * f * n_theta_batch_row * sizeof(dtype)));

			#ifdef MEASURE_PERF
				printf("****** Get Hermitian theta (batch %d / %d)\n", batch_id + 1, n_theta_batch);
				t1 = seconds();
			#endif

			//if(f == 100){
			//	get_hermitian100<<<batch_size, 64, SCAN_BATCH * f/2*sizeof(dtype2)>>>
			//		(batch_offset, (dtype2*)xx, cscColIndex, cscRowIndex, lambda, n, f, (dtype2*)XT);
			//}
			//else
				get_hermitianT10<<<n_theta_batch_row, dim_a_tile, SCAN_BATCH * f * sizeof(dtype)>>>
					(n_theta_batch_offset_row, d_b_array, d_csc_CI, d_csc_RI, lambda, 0., n, f, d_xT);
			cudaDeviceSynchronize();
			cudaCheckError();
			#ifdef MEASURE_PERF
				printf("****** Get Hermitian kernel (batch %d / %d) runs %f seconds.\n", batch_id + 1, n_theta_batch, seconds() - t1);
			#endif			

			#ifdef MEASURE_PERF
				printf("****** Update theta (batch %d / %d)\n", batch_id + 1, n_theta_batch);
				t1 = seconds();
			#endif
			#ifdef USE_CG
				updateXWithCGHost(xx, &thetaT[batch_offset*f], &yTXT[batch_offset*f], batch_size, f, CG_ITER);
			#else
				dtype **d_b_p_array = 0; // original: devPtrXXHost
				dtype **d_yTxT_p_array = 0; // original: devPtrYTXTHost
				cudacall(cudaMallocHost((void**)&d_b_p_array, n_theta_batch_row * sizeof(*d_b_p_array)));
				cudacall(cudaMallocHost((void**)&d_yTxT_p_array, n_theta_batch_row * sizeof(*d_yTxT_p_array)));
				updateTheta(n_theta_batch_row, n_theta_batch_offset_row, d_yTxT, d_b_array, d_thetaT, cublas_handle, m,  n,  f,  nnz, d_b_p_array, d_yTxT_p_array);
				cudacall(cudaFreeHost(d_b_p_array));
				cudacall(cudaFreeHost(d_yTxT_p_array));
			#endif
			#ifdef MEASURE_PERF
				printf("****** Update theta solver (batch %d / %d) runs seconds: %f \n", batch_id + 1, n_theta_batch, seconds() - t1);
			#endif
			cudacall(cudaFree(d_b_array));
		}
		#ifdef MEASURE_PERF
			printf("****** Update theta runs %f seconds, gridSize: %d, blockSize %d.\n", seconds() - t0, n, f);
		#endif
		cudacall(cudaFree(d_yTxT));

		#ifdef MEASURE_PERF
			printf("------------------------------ Calculate MAE ------------------------------\n");
		#endif
		int n_error_bucket = 1000; //original: error_size
		int n_error_thread = 256;

		dtype *error_train_bucket = 0;
		cudacall(cudaMalloc((void**)&error_train_bucket, n_error_bucket * sizeof(error_train_bucket[0])));
		cudacall(cudaMemset(error_train_bucket, 0, n_error_bucket * sizeof(dtype)) );

		cudacall(cudaMalloc((void**)&d_coo_RI, nnz * sizeof(d_coo_RI[0])));
		cudacall(cudaMalloc((void**)&d_csr_CI, nnz * sizeof(d_csr_CI[0])));
		cudacall(cudaMalloc((void**)&d_csr_Val, nnz * sizeof(d_csr_Val[0])));
		cudacall(cudaMemcpy(d_coo_RI, h_coo_RI, (size_t)(nnz * sizeof(d_coo_RI[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(d_csr_CI, h_csr_CI, (size_t)(nnz * sizeof(d_csr_CI[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(d_csr_Val, h_csr_Val, (size_t)(nnz * sizeof(d_csr_Val[0])), cudaMemcpyHostToDevice));

		MAE<<<(nnz - 1) / n_error_thread + 1, n_error_thread>>>
				(d_csr_Val, d_coo_RI, d_csr_CI, d_thetaT, d_xT, error_train_bucket, nnz, n_error_bucket, f);
		cudaDeviceSynchronize();
		cudaCheckError();
		cudacall(cudaFree(d_coo_RI));
		cudacall(cudaFree(d_csr_CI));
		cudacall(cudaFree(d_csr_Val));

		// reduce traning error
		dtype* error_train = 0;
		cudacall(cudaMallocHost((void**)&error_train, sizeof(dtype)));
#ifdef USE_DOUBLE
		cublascall(cublasDasum(cublas_handle, n_error_bucket, error_train_bucket, 1, error_train));
#else
		cublascall(cublasSasum(cublas_handle, n_error_bucket, error_train_bucket, 1, error_train));
#endif
		cudaDeviceSynchronize();
		cudaCheckError();
		*error_train /= nnz;
		if (verbose) printf("****** Training error in iter %d: %.12f\n", iter, *error_train);
		cudacall(cudaFree(error_train_bucket));

		
		dtype *error_test_bucket = 0;
		cudacall(cudaMalloc((void**)&error_test_bucket, n_error_bucket * sizeof(error_test_bucket[0])));
		cudacall(cudaMemset(error_test_bucket, 0, n_error_bucket * sizeof(dtype)));

		cudacall(cudaMalloc((void**)&d_test_coo_RI, nnz_test * sizeof(d_test_coo_RI[0])));
		cudacall(cudaMalloc((void**)&d_test_coo_CI, nnz_test * sizeof(d_test_coo_CI[0])));
		cudacall(cudaMalloc((void**)&d_test_coo_Val, nnz_test * sizeof(d_test_coo_Val[0])));
		cudacall(cudaMemcpy(d_test_coo_RI, h_test_coo_RI, (size_t)(nnz_test * sizeof(d_test_coo_RI[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(d_test_coo_CI, h_test_coo_CI, (size_t)(nnz_test * sizeof(d_test_coo_CI[0])), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(d_test_coo_Val, h_test_coo_Val, (size_t)(nnz_test * sizeof(d_test_coo_Val[0])), cudaMemcpyHostToDevice));

		MAE<<<(nnz_test - 1) / n_error_thread + 1, n_error_thread>>>
				(d_test_coo_Val, d_test_coo_RI, d_test_coo_CI, d_thetaT, d_xT, error_test_bucket, nnz_test, n_error_bucket, f);
		cudaDeviceSynchronize();
		cudaCheckError();
		cudacall(cudaFree(d_test_coo_RI));
		cudacall(cudaFree(d_test_coo_CI));
		cudacall(cudaFree(d_test_coo_Val));

		dtype* error_test = 0;
		cudacall(cudaMallocHost((void**)&error_test, sizeof(dtype)));
#ifdef USE_DOUBLE
		cublascall(cublasDasum(cublas_handle, n_error_bucket, error_test_bucket, 1, error_test));
#else
		cublascall(cublasSasum(cublas_handle, n_error_bucket, error_test_bucket, 1, error_test));
#endif
		cudaDeviceSynchronize();
		*error_test /= nnz_test;
		if (verbose) printf("****** Testing error in iter %d: %.12f\n", iter, *error_test);
		ret_error_test = *error_test;
		cudacall(cudaFree(error_test_bucket));
	}

	// copy feature vectors back to host
	cudacall(cudaMemcpy(h_thetaT, d_thetaT, (size_t)(n * f * sizeof(h_thetaT[0])), cudaMemcpyDeviceToHost));
	cudacall(cudaMemcpy(h_xT, d_xT, (size_t)(m * f * sizeof(h_xT[0])), cudaMemcpyDeviceToHost));
	cudacall(cudaFree(d_thetaT));
	cudacall(cudaFree(d_xT));
	cudacall(cudaFree(d_csc_Val));
	cudacall(cudaFree(d_csc_CI));
	cudacall(cudaFree(d_csc_RI));

	//WARN: do not call cudaDeviceReset inside ALS() 
	//because the caller needs to access XT and thetaT which was in the same context
	//cudacall(cudaDeviceReset());
	return ret_error_test;
}
