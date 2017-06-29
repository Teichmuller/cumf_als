/*
 * main.cpp
 *
 *  Created on: Feb 10, 2015
 *      Author: Wei Tan (wtan@us.ibm.com)
 *  Test als.cu using netflix or yahoo data
 *  Alternating Least Square for Matrix Factorization on CUDA 7.0+
 *  Code optimized for F = 100, and on cc 3.5, 3.7 platforms. Also tested in cc 5.2
 */
#include "als.h"
#include "host_utilities.h"
#include<stdlib.h>
#include<stdio.h>
#include <string>

#define DEVICEID 0
#define ITERS 1

int main(int argc, char **argv) {
	//parse input parameters
	if(argc != 10){
		printf("Usage: give M, N, F, NNZ, NNZ_TEST, lambda, X_BATCH, THETA_BATCH and DATA_DIR.\n");
		printf("E.g., for netflix data set, use: \n");
		printf("./main 17770 480189 100 99072112 1408395 0.048 1 3 ./data/netflix/ \n");
		printf("E.g., for movielens 10M data set, use: \n");
		printf("./main 71567 65133 100 9000048 1000006 0.05 1 1 ./data/ml10M/ \n");
		printf("E.g., for yahooMusic data set, use: \n");
		printf("./main 1000990 624961 100 252800275 4003960 1.4 6 3 ./data/yahoo/ \n");
		return 0;
	}
	
	int f = atoi(argv[3]);
	if(f%T10!=0){
		printf("F has to be a multiple of %d \n", T10);
		return 0;
	}
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	long nnz = atoi(argv[4]);
	long nnz_test = atoi(argv[5]);
	dtype lambda = atof(argv[6]);
	int X_BATCH = atoi(argv[7]);
	int THETA_BATCH = atoi(argv[8]);
	std::string DATA_DIR(argv[9]);
	printf("M = %d, N = %d, F = %d, NNZ = %ld, NNZ_TEST = %ld, lambda = %f\nX_BATCH = %d, THETA_BATCH = %d\nDATA_DIR = %s \n",
			m, n, f, nnz, nnz_test, lambda, X_BATCH, THETA_BATCH, DATA_DIR.c_str());
	
	cudaSetDevice(DEVICEID);
	int* csrRowIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrRowIndexHostPtr, (m + 1) * sizeof(csrRowIndexHostPtr[0])) );
	int* csrColIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrColIndexHostPtr, nnz * sizeof(csrColIndexHostPtr[0])) );
	dtype* csrValHostPtr;
	cudacall(cudaMallocHost( (void** ) &csrValHostPtr, nnz * sizeof(csrValHostPtr[0])) );
	dtype* cscValHostPtr;
	cudacall(cudaMallocHost( (void** ) &cscValHostPtr, nnz * sizeof(cscValHostPtr[0])) );
	int* cscRowIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &cscRowIndexHostPtr, nnz * sizeof(cscRowIndexHostPtr[0])) );
	int* cscColIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &cscColIndexHostPtr, (n+1) * sizeof(cscColIndexHostPtr[0])) );
	int* cooRowIndexHostPtr;
	cudacall(cudaMallocHost( (void** ) &cooRowIndexHostPtr, nnz * sizeof(cooRowIndexHostPtr[0])) );

	//calculate X from thetaT first, need to initialize thetaT
	dtype* thetaTHost;
	cudacall(cudaMallocHost( (void** ) &thetaTHost, n * f * sizeof(thetaTHost[0])) );

	dtype* XTHost;
	cudacall(cudaMallocHost( (void** ) &XTHost, m * f * sizeof(XTHost[0])) );

	/* But I comment out here to use same initial values as test_case2			----Xuan
	//initialize thetaT on host
	unsigned int seed = 0;
	srand (seed);
	for (int k = 0; k < n * f; k++)
		thetaTHost[k] = 0.2*((dtype) rand() / (dtype)RAND_MAX);
	//CG needs to initialize X as well
	for (int k = 0; k < m * f; k++)
		XTHost[k] = 0;//0.1*((dtype) rand() / (dtype)RAND_MAX);;

	*/
	FILE * xinitfile = fopen((DATA_DIR + "/XT.init.bin").c_str(), "rb");
	FILE * thetainitfile = fopen((DATA_DIR + "thetaT.init.bin").c_str(), "rb");
	float* thetaTHost_tmp;
	float* XTHost_tmp;
	cudacall(cudaMallocHost( (void** ) &thetaTHost_tmp, n * f * sizeof(thetaTHost_tmp[0])) );
	cudacall(cudaMallocHost( (void** ) &XTHost_tmp, m * f * sizeof(XTHost_tmp[0])) );
	fread(XTHost_tmp, sizeof(dtype), m*f, xinitfile);
	fread(thetaTHost_tmp, sizeof(dtype), n*f, thetainitfile);
	fclose(xinitfile);
	fclose(thetainitfile);
	for (int i = 0; i < m * f; i++)
		XTHost[i] = XTHost_tmp[i];
	for (int i = 0; i < n * f; i++)
		thetaTHost[i] = thetaTHost_tmp[i];
	cudaFreeHost(XTHost_tmp);
	cudaFreeHost(thetaTHost_tmp);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	printf("*******start loading training and testing sets to host.\n");
	//testing set
	int* cooRowIndexTestHostPtr = (int *) malloc(
			nnz_test * sizeof(cooRowIndexTestHostPtr[0]));
	int* cooColIndexTestHostPtr = (int *) malloc(
			nnz_test * sizeof(cooColIndexTestHostPtr[0]));
	dtype* cooValHostTestPtr = (dtype *) malloc(nnz_test * sizeof(cooValHostTestPtr[0]));

	struct timeval tv0;
	gettimeofday(&tv0, NULL);

	float* cooValHostTestPtr_tmp;
	cudacall(cudaMallocHost( (void** ) &cooValHostTestPtr_tmp, nnz * sizeof(cooValHostTestPtr_tmp[0])) );
	loadCooSparseMatrixBin( (DATA_DIR + "/R_test_coo.data.bin").c_str(), (DATA_DIR + "/R_test_coo.row.bin").c_str(), 
							(DATA_DIR + "/R_test_coo.col.bin").c_str(),
			cooValHostTestPtr_tmp, cooRowIndexTestHostPtr, cooColIndexTestHostPtr, nnz_test);
    for (int i = 0; i < nnz; i++) cooValHostTestPtr[i] = cooValHostTestPtr_tmp[i];
	cudaFreeHost(cooValHostTestPtr_tmp);


	float* csrValHostPtr_tmp;
	cudacall(cudaMallocHost( (void** ) &csrValHostPtr_tmp, nnz * sizeof(csrValHostPtr_tmp[0])) );
    loadCSRSparseMatrixBin( (DATA_DIR + "/R_train_csr.data.bin").c_str(), (DATA_DIR + "/R_train_csr.indptr.bin").c_str(),
							(DATA_DIR + "/R_train_csr.indices.bin").c_str(),
    		csrValHostPtr_tmp, csrRowIndexHostPtr, csrColIndexHostPtr, m, nnz);
    for (int i = 0; i < nnz; i++) csrValHostPtr[i] = csrValHostPtr_tmp[i];
	cudaFreeHost(csrValHostPtr_tmp);

	float* cscValHostPtr_tmp;
	cudacall(cudaMallocHost( (void** ) &cscValHostPtr_tmp, nnz * sizeof(cscValHostPtr_tmp[0])) );
    loadCSCSparseMatrixBin( (DATA_DIR + "/R_train_csc.data.bin").c_str(), (DATA_DIR + "/R_train_csc.indices.bin").c_str(),
							(DATA_DIR +"/R_train_csc.indptr.bin").c_str(),
   		cscValHostPtr_tmp, cscRowIndexHostPtr, cscColIndexHostPtr, n, nnz);
    for (int i = 0; i < nnz; i++) cscValHostPtr[i] = cscValHostPtr_tmp[i];
	cudaFreeHost(cscValHostPtr_tmp);

    loadCooSparseMatrixRowPtrBin( (DATA_DIR + "/R_train_coo.row.bin").c_str(), cooRowIndexHostPtr, nnz);
	


	#ifdef DEBUG
    printf("\nloaded training csr to host; print data, row and col array\n");
	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%.1f ", csrValHostPtr[i]);
	}
	printf("\n");

	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%d ", csrRowIndexHostPtr[i]);
	}
	printf("\n");
	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%d ", csrColIndexHostPtr[i]);
	}
	printf("\n");
	
	printf("\nloaded testing coo to host; print data, row and col array\n");
	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%.1f ", cooValHostTestPtr[i]);
	}
	printf("\n");

	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%d ", cooRowIndexTestHostPtr[i]);
	}
	printf("\n");
	for (int i = 0; i < nnz && i < 10; i++) {
		printf("%d ", cooColIndexTestHostPtr[i]);
	}
	printf("\n");
	
	#endif
	double t0 = seconds();
	
	doALS(csrRowIndexHostPtr, csrColIndexHostPtr, csrValHostPtr,
			cscRowIndexHostPtr, cscColIndexHostPtr, cscValHostPtr,
			cooRowIndexHostPtr, thetaTHost, XTHost,
			cooRowIndexTestHostPtr, cooColIndexTestHostPtr, cooValHostTestPtr,
			m, n, f, nnz, nnz_test, lambda,
			ITERS, X_BATCH, THETA_BATCH, DEVICEID);
	printf("\ndoALS takes seconds: %.3f for F = %d\n", seconds() - t0, f);


	//write out the model	
	FILE * xfile = fopen((DATA_DIR + "XT.data.bin").c_str(), "wb");
	FILE * thetafile = fopen((DATA_DIR + "thetaT.data.bin").c_str(), "wb");
	fwrite(XTHost, sizeof(dtype), m*f, xfile);
	fwrite(thetaTHost, sizeof(dtype), n*f, thetafile);
	fclose(xfile);
	fclose(thetafile);
	


	cudaFreeHost(csrRowIndexHostPtr);
	cudaFreeHost(csrColIndexHostPtr);
	cudaFreeHost(csrValHostPtr);
	cudaFreeHost(cscValHostPtr);
	cudaFreeHost(cscRowIndexHostPtr);
	cudaFreeHost(cscColIndexHostPtr);
	cudaFreeHost(cooRowIndexHostPtr);
	cudaFreeHost(XTHost);
	cudaFreeHost(thetaTHost);
	cudacall(cudaDeviceReset());
	printf("\nALS Done.\n");
	return 0;
}
