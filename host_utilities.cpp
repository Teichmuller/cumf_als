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
#include "host_utilities.h"
#include <fstream>

bool loadCSRSparseMatrixBin(const std::string& dataFile, const std::string& rowFile, const std::string& colFile,
		float* data, int* row, int* col, const int m, const long nnz) {
	#ifdef DEBUG
    printf("\n loading CSR...\n");
	#endif
	FILE *dFile = fopen(dataFile.data(), "rb");
	FILE *rFile = fopen(rowFile.data(), "rb");
	FILE *cFile = fopen(colFile.data(), "rb");

	CHECK_MSG_RET_FALSE(dFile, "Unable to open file " << dataFile << "!")
	CHECK_MSG_RET_FALSE(rFile, "Unable to open file " << rowFile << "!")
	CHECK_MSG_RET_FALSE(cFile, "Unable to open file " << colFile << "!")

	CHECK_MSG_RET_FALSE(fread(&data[0], sizeof(data[0]), nnz, dFile) == nnz, "Loading Val failed!")
	CHECK_MSG_RET_FALSE(fread(&row[0], sizeof(row[0]), m + 1, rFile) == m + 1, "Loading RI failed!")
	CHECK_MSG_RET_FALSE(fread(&col[0], sizeof(col[0]), nnz, cFile) == nnz, "Loading CI failed!")

	fclose(rFile);
	fclose(dFile);
	fclose(cFile);
	return true;
}

bool loadCSCSparseMatrixBin(const std::string& dataFile, const std::string& rowFile, const std::string& colFile,
		float * data, int* row, int* col, const int n, const long nnz) {
	#ifdef DEBUG		
    printf("\n loading CSC...\n");
	#endif
	FILE *dFile = fopen(dataFile.data(), "rb");
	FILE *rFile = fopen(rowFile.data(), "rb");
	FILE *cFile = fopen(colFile.data(), "rb");

	CHECK_MSG_RET_FALSE(dFile, "Unable to open file " << dataFile << "!")
	CHECK_MSG_RET_FALSE(rFile, "Unable to open file " << rowFile << "!")
	CHECK_MSG_RET_FALSE(cFile, "Unable to open file " << colFile << "!")

	CHECK_MSG_RET_FALSE(fread(&data[0], sizeof(data[0]), nnz, dFile) == nnz, "Loading Val failed!")
	CHECK_MSG_RET_FALSE(fread(&row[0], sizeof(row[0]), nnz, rFile) == nnz, "Loading RI failed!")
	CHECK_MSG_RET_FALSE(fread(&col[0], sizeof(col[0]), n + 1, cFile) == n + 1, "Loading CI failed!")

	fclose(dFile);
	fclose(rFile);
	fclose(cFile);
	return true;
}

bool loadCooSparseMatrixRowPtrBin(const std::string& rowFile, int* row, const long nnz) {
	#ifdef DEBUG
    printf("\n loading COO Row...\n");
	#endif
	FILE *rFile = fopen(rowFile.data(), "rb");
	CHECK_MSG_RET_FALSE(fread(&row[0], sizeof(row[0]), nnz, rFile) == nnz, "Loading RI failed!")
	fclose(rFile);
	return true;
}

bool loadCooSparseMatrixBin(const std::string& dataFile, const std::string& rowFile, const std::string& colFile,
		float* data, int* row, int* col, const long nnz) {
	#ifdef DEBUG
    printf("\n loading COO...\n");
	#endif
	FILE *dFile = fopen(dataFile.data(), "rb");
	FILE *rFile = fopen(rowFile.data(), "rb");
	FILE *cFile = fopen(colFile.data(), "rb");

	CHECK_MSG_RET_FALSE(dFile, "Unable to open file " << dataFile << "!")
	CHECK_MSG_RET_FALSE(rFile, "Unable to open file " << rowFile << "!")
	CHECK_MSG_RET_FALSE(cFile, "Unable to open file " << colFile << "!")

	CHECK_MSG_RET_FALSE(fread(&data[0], sizeof(data[0]), nnz, dFile) == nnz, "Loading Val failed!")
	CHECK_MSG_RET_FALSE(fread(&row[0], sizeof(row[0]), nnz, rFile) == nnz, "Loading RI failed!")
	CHECK_MSG_RET_FALSE(fread(&col[0], sizeof(col[0]), nnz, cFile) == nnz, "Loading CI failed!")

	fclose(dFile);
	fclose(rFile);
	fclose(cFile);
	return true;
}

