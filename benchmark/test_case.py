import pandas as pd
import numpy as np
#import matplotlib.pyplot as plot
import collections as cl
import time
from config import *
from utils import *
from IALS_Tensorflow import *


def test_case(name, rank, Lambda, Mu, MaxIter):

	test_case_path = data_path + name + system_sep
	data_file = test_case_path + 'training_data.npy'
	truth_file = test_case_path + 'ground_truth.npy'

	print('Loading data...')
	data, truth, init_y, init_x = recover_test_case(name, rank)

	print('Running modidiedals...')
	x, y, train_mae, test_mae, elapse = modifiedals_noinc(data, rank, truth,
										lam = Lambda, mu = Mu, thres = 0.0000, maxit = MaxIter + 1,
										init_x = init_x.transpose(), init_y = init_y.transpose(), dtype = np.float64)
	res = np.matmul(x.transpose(), y)
	print('Numpy elapse: ', elapse)
	print('MALS training MAE: ', train_mae)
	print('MALS testing MAE: ', test_mae)

	print('TF ALS preprocessing...')
	data_mask = np.transpose(~np.isnan(data))
	data_processed = np.where(np.transpose(data_mask), data, np.zeros(data.shape)).transpose()
	als = ALS(rank, Lambda, Mu, MaxIter = MaxIter, tf_Dtype = tf.float64)
	als.UpdateParallelIter = 2048
	print('Running TF ALS...')
	start_time = time.time()
	r, u, v = als.Run(data_processed, data_mask, init_y, init_x)
	end_time = time.time()
	print('Total Time: ', end_time - start_time)

	train_mae_tf = train_MAE(r, data_mask, truth)
	print('TF training MAE: ', train_mae_tf)
	test_mae_tf = test_MAE(r, data_mask, truth)
	print('TF testing MAE: ', test_mae_tf)

	x_gpu = read_binary_fp64_file(test_case_path + 'XT.data.bin', [data.shape[1], rank])
	theta_gpu = read_binary_fp64_file(test_case_path + 'thetaT.data.bin', [data.shape[0], rank])

	csc_indptr = read_binary_int32_file(test_case_path + 'R_train_csc.indptr.bin', None)
	csc_indices = read_binary_int32_file(test_case_path + 'R_train_csc.indices.bin', None)
	csc_val = read_binary_fp32_file(test_case_path + 'R_train_csc.data.bin', None)
	coo_test_col = read_binary_int32_file(test_case_path + 'R_test_coo.col.bin', None)
	coo_test_row = read_binary_int32_file(test_case_path + 'R_test_coo.row.bin', None)
	coo_test_val = read_binary_fp32_file(test_case_path + 'R_test_coo.data.bin', None)

	eps = 1e-6
	print('Is there any error beyond tolerance? ', np.any(np.abs(x_gpu - y.T) > eps))
	print('Is there any error beyond tolerance? ', np.any(np.abs(x_gpu - u) > eps))
	print('Is there any error beyond tolerance? ', np.any(np.abs(theta_gpu - x.T) > eps))
	print('Is there any error beyond tolerance? ', np.any(np.abs(theta_gpu - v) > eps))
