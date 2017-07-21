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
#include <stdlib.h>
#include <stdio.h>
#include <string>

#define DEVICE_ID 0

int main(int argc, char **argv)
{
	//parse input parameters
	if (argc != 12)
	{
		printf("Usage: give M, N, F, NNZ, NNZ_TEST, lambda, mu, iter, X_BATCH, THETA_BATCH and DATA_DIR.\n");
		printf("E.g. 96 1000000 20 17171442 22426274 3.0 2.0 10 1 1 ./data/test_case3\n");
		printf("E.g. 96 1000 20 30234 50607 3.0 2.0 10 1 1 ./data/test_case3_1000");
	}
	
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int f = atoi(argv[3]);
	long nnz = atoi(argv[4]);
	long nnz_test = atoi(argv[5]);
	dtype lambda = atof(argv[6]);
	dtype mu = atof(argv[7]);
	int iter = atoi(argv[8]);
	int n_x_batch = atoi(argv[9]);
	int n_theta_batch = atoi(argv[10]);
	std::string data_dir(argv[11]);

	if (f % RegisterTileSize != 0)
	{
		printf("F has to be a multiple of %d \n", RegisterTileSize);
		return 0;
	}
	printf("M = %d, N = %d, F = %d, NNZ = %ld, NNZ_TEST = %ld, lambda = %f, mu = %f\nX_BATCH = %d, THETA_BATCH = %d\nDATA_DIR = %s \n",
			m, n, f, nnz, nnz_test, lambda, mu, n_x_batch, n_theta_batch, data_dir.c_str());
	
	cudaSetDevice(DEVICE_ID);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	printf("****** Loading Initial Value ******\n");

	// calculate X from thetaT first, need to initialize thetaT

	/* But I comment out here to use same initial values as other tests			----Xuan
	//initialize thetaT on host
	unsigned int seed = 0;
	srand (seed);
	for (int k = 0; k < n * f; k++)
		thetaTHost[k] = 0.2*((dtype) rand() / (dtype)RAND_MAX);
	//CG needs to initialize X as well
	for (int k = 0; k < m * f; k++)
		XTHost[k] = 0;//0.1*((dtype) rand() / (dtype)RAND_MAX);;

	*/
	float* h_xT_tmp;
	float* h_thetaT_tmp;
	cudacall(cudaMallocHost((void**)&h_xT_tmp, m * f * sizeof(h_xT_tmp[0])));
	cudacall(cudaMallocHost((void**)&h_thetaT_tmp, n * f * sizeof(h_thetaT_tmp[0])));
	FILE * f_xT_init = fopen((data_dir + "/XT.init.bin").c_str(), "rb");
	FILE * f_thetaT_init = fopen((data_dir + "/thetaT.init.bin").c_str(), "rb");
		CHECK_MSG_FINAL(fread(h_xT_tmp, sizeof(h_xT_tmp[0]), m * f, f_xT_init) == m * f, "Reading xT_init failed!",
				{
						fclose(f_xT_init);
						return 0;
				}
		)
	fclose(f_xT_init);
		CHECK_MSG_FINAL(fread(h_thetaT_tmp, sizeof(h_thetaT_tmp[0]), n * f, f_thetaT_init) == n * f, "Reading thetaT_init failed!",
				{
						fclose(f_thetaT_init);
						return 0;
				}
		)
	fclose(f_thetaT_init);

	dtype* h_xT;
	dtype* h_thetaT;
	cudacall(cudaMallocHost((void**)&h_xT, m * f * sizeof(h_xT[0])));
	cudacall(cudaMallocHost((void**)&h_thetaT, n * f * sizeof(h_thetaT[0])));
	for (int i = 0; i < m * f; i++)
		h_xT[i] = h_xT_tmp[i];
	for (int i = 0; i < n * f; i++)
		h_thetaT[i] = h_thetaT_tmp[i];
	cudaFreeHost(h_xT_tmp);
	cudaFreeHost(h_thetaT_tmp);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	printf("****** Loading Data Sets ******\n");

	// training set
	int* h_csr_RI;
	int* h_csr_CI;
	int* h_csc_RI;
	int* h_csc_CI;
	int* h_coo_RI;
	dtype* h_csr_Val;
	dtype* h_csc_Val;
	cudacall(cudaMallocHost((void**)&h_csr_RI, (m + 1) * sizeof(h_csr_RI[0])));
	cudacall(cudaMallocHost((void**)&h_csc_CI, (n + 1) * sizeof(h_csc_CI[0])));
	cudacall(cudaMallocHost((void**)&h_csr_CI, nnz * sizeof(h_csr_CI[0])));
	cudacall(cudaMallocHost((void**)&h_csc_RI, nnz * sizeof(h_csc_RI[0])));
	cudacall(cudaMallocHost((void**)&h_coo_RI, nnz * sizeof(h_coo_RI[0])));
	cudacall(cudaMallocHost((void**)&h_csr_Val, nnz * sizeof(h_csr_Val[0])));
	cudacall(cudaMallocHost((void**)&h_csc_Val, nnz * sizeof(h_csc_Val[0])));
	
	// testing set
	int* h_test_coo_RI;
	int* h_test_coo_CI;
	dtype* h_test_coo_Val;
	cudacall(cudaMallocHost((void**)&h_test_coo_RI, nnz_test * sizeof(h_test_coo_RI[0])));
	cudacall(cudaMallocHost((void**)&h_test_coo_CI, nnz_test * sizeof(h_test_coo_CI[0])));
	cudacall(cudaMallocHost((void**)&h_test_coo_Val, nnz_test * sizeof(h_test_coo_Val[0])));

	// load
	float* h_csr_Val_tmp;
	float* h_csc_Val_tmp;
	float* h_test_coo_Val_tmp;
	// loading testing set
	cudacall(cudaMallocHost((void**)&h_test_coo_Val_tmp, nnz_test * sizeof(h_test_coo_Val_tmp[0])));
	loadCooSparseMatrixBin(data_dir + "/R_test_coo.data.bin", data_dir + "/R_test_coo.row.bin", data_dir + "/R_test_coo.col.bin",
			h_test_coo_Val_tmp, h_test_coo_RI, h_test_coo_CI, nnz_test);
    for (int i = 0; i < nnz_test; i++) h_test_coo_Val[i] = h_test_coo_Val_tmp[i];
	cudaFreeHost(h_test_coo_Val_tmp);
	// loading training set in csr
	cudacall(cudaMallocHost((void**)&h_csr_Val_tmp, nnz * sizeof(h_csr_Val_tmp[0])));
    loadCSRSparseMatrixBin(data_dir + "/R_train_csr.data.bin", data_dir + "/R_train_csr.indptr.bin", data_dir + "/R_train_csr.indices.bin",
    		h_csr_Val_tmp, h_csr_RI, h_csr_CI, m, nnz);
    for (int i = 0; i < nnz; i++) h_csr_Val[i] = h_csr_Val_tmp[i];
	cudaFreeHost(h_csr_Val_tmp);
	// loading training set in csc
	cudacall(cudaMallocHost((void**)&h_csc_Val_tmp, nnz * sizeof(h_csc_Val_tmp[0])) );
    loadCSCSparseMatrixBin(data_dir + "/R_train_csc.data.bin", data_dir + "/R_train_csc.indices.bin", data_dir + "/R_train_csc.indptr.bin",
    		h_csc_Val_tmp, h_csc_RI, h_csc_CI, n, nnz);
    for (int i = 0; i < nnz; i++) h_csc_Val[i] = h_csc_Val_tmp[i];
	cudaFreeHost(h_csc_Val_tmp);
	// loading trainig set in coo
    loadCooSparseMatrixRowPtrBin(data_dir + "/R_train_coo.row.bin", h_coo_RI, nnz);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	printf("****** Running ALS ******\n");

    // DO!
	double t0 = seconds();
	doALS(h_csr_RI, h_csr_CI, h_csr_Val,
		h_csc_RI, h_csc_CI, h_csc_Val,
		h_coo_RI, h_thetaT, h_xT,
		h_test_coo_RI, h_test_coo_CI, h_test_coo_Val,
		m, n, f, nnz, nnz_test, lambda, mu,
		iter, n_x_batch, n_theta_batch, DEVICE_ID, true);
	printf("ALS takes %.3f seconds\n", seconds() - t0);

	////////////////////////////////////////////////////////////////////////////////////////////////////
	printf("****** Saving Model ******\n");

	// write out the model
	FILE* f_xT = fopen((data_dir + "/XT.data.bin").c_str(), "wb");
	CHECK_MSG_FINAL(fwrite(h_xT, sizeof(h_xT[0]), m * f, f_xT) == m * f, "Writing xT failed!",
			{
					fclose(f_xT);
					return 0;
			}
	)
	fclose(f_xT);
	FILE* f_thetaT = fopen((data_dir + "/thetaT.data.bin").c_str(), "wb");
	CHECK_MSG_FINAL(fwrite(h_thetaT, sizeof(h_thetaT[0]), n * f, f_thetaT) == n * f, "Writing thetaT failed!",
			{
					fclose(f_thetaT);
					return 0;
			}
	)
	fclose(f_thetaT);

	cudaFreeHost(h_csr_RI);
	cudaFreeHost(h_csr_CI);
	cudaFreeHost(h_csr_Val);
	cudaFreeHost(h_csc_RI);
	cudaFreeHost(h_csc_CI);
	cudaFreeHost(h_csc_Val);
	cudaFreeHost(h_coo_RI);
	cudaFreeHost(h_xT);
	cudaFreeHost(h_thetaT);
	cudacall(cudaDeviceReset());

	////////////////////////////////////////////////////////////////////////////////////////////////////
	printf("\nALS Done.\n");
	return 0;
}
