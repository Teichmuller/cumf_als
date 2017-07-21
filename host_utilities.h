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
 * define some host utility functions, 
 * such as timing and data loading (to host memory)
 */
#ifndef HOST_UTILITIES_H_
#define HOST_UTILITIES_H_
#include <sys/time.h>
#include <iostream>
#include <string>

inline double seconds(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

bool loadCSRSparseMatrixBin(const std::string& dataFile, const std::string& rowFile, const std::string& colFile,
		float* data, int* row, int* col, const int m, const long nnz);

bool loadCSCSparseMatrixBin(const std::string& dataFile, const std::string& rowFile, const std::string& colFile,
		float * data, int* row, int* col, const int n, const long nnz);

bool loadCooSparseMatrixRowPtrBin(const std::string& rowFile, int* row, const long nnz);

bool loadCooSparseMatrixBin(const std::string& dataFile, const std::string& rowFile, const std::string& colFile,
		float* data, int* row, int* col, const long nnz);


#define MSG(msg)                            {std::cout << "ERROR: " << __FILE__ << " (line: " << __LINE__ << ") in " << __func__ << " :\n" << msg << std::endl;}

#define MSG_FINAL(msg, act)                 {MSG(msg) {act}}

#define CHECK_MSG(exp, msg)                 {if (!(exp))    {MSG(msg)}}
#define CHECK_MSG_FINAL(exp, msg, act)      {if (!(exp))    {MSG_FINAL(msg, act)}}

#define CHECK_MSG_RET(exp, msg, val)        CHECK_MSG_FINAL(exp, msg, return (val);)
#define CHECK_MSG_RET_TRUE(exp, msg)        CHECK_MSG_RET(exp, msg, true)
#define CHECK_MSG_RET_FALSE(exp, msg)       CHECK_MSG_RET(exp, msg, false)
#define CHECK_MSG_RET_NULLPTR(exp, msg)     CHECK_MSG_RET(exp, msg, nullptr)

#endif
