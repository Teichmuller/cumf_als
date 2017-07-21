import time
import pandas as pd
from six.moves import urllib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy import sparse
from config import *

	#data_file = exchange_data_folder + 'excavated_data.npy'
	#truth_file = exchange_data_folder + 'data.npy'

def make_test_case(name, data_, truth_, rank, u_init = None, v_init = None, dtype = np.float32):

	test_case_path = data_path + name + system_sep
	data_file = test_case_path + 'training_data.npy'
	truth_file = test_case_path + 'ground_truth.npy'

	np.save(data_file, data_)
	np.save(truth_file, truth_)

	data = data_.T
	truth = truth_.T
	m = data.shape[0]
	n = data.shape[1]

	# to-format explaination
	"""
	0 0 83.34
	0 1 45.21
	5 6 43.65
	x y val
	"""
	data_mask = ~np.isnan(data)
	truth_mask = ~np.isnan(truth)
	train_mask = np.logical_and(truth_mask, data_mask)
	test_mask = np.logical_and(truth_mask, np.logical_not(data_mask))
	train_loc = np.where(train_mask)
	test_loc = np.where(test_mask)
	train_val = truth[train_loc]
	test_val = truth[test_loc]
	train_loc = np.vstack((train_loc[0], train_loc[1])).T
	test_loc = np.vstack((test_loc[0], test_loc[1])).T
	# In[7]:

	user_item_train, user_item_test, rating_train, rating_test = (train_loc, test_loc, train_val, test_val)
	nnz_train = np.where(train_mask, np.ones(train_mask.shape, np.int), np.zeros(train_mask.shape, np.int)).sum()
	nnz_test = np.where(test_mask, np.ones(test_mask.shape, np.int), np.zeros(test_mask.shape, np.int)).sum()


	# In[8]:

	#for test data, we need COO format to calculate test RMSE
	#1-based to 0-based
	R_test_coo = coo_matrix((rating_test,(user_item_test[:,0],user_item_test[:,1])))
	assert R_test_coo.nnz == nnz_test
	R_test_coo.data.astype(dtype).tofile(test_case_path + 'R_test_coo.data.bin')
	R_test_coo.row.tofile(test_case_path + 'R_test_coo.row.bin')
	R_test_coo.col.tofile(test_case_path + 'R_test_coo.col.bin')


	# In[9]:

	print(np.max(R_test_coo.data))
	print(np.max(R_test_coo.row))
	print(np.max(R_test_coo.col))
	print(R_test_coo.data)
	print(R_test_coo.row)
	print(R_test_coo.col)


	# In[10]:

	test_data = np.fromfile(test_case_path + 'R_test_coo.data.bin',dtype=dtype)
	test_row = np.fromfile(test_case_path + 'R_test_coo.row.bin', dtype=np.int32)
	test_col = np.fromfile(test_case_path + 'R_test_coo.col.bin',dtype=np.int32)
	print(test_data[0:10])
	print(test_row[0:10])
	print(test_col[0:10])


	# In[11]:

	#1-based to 0-based
	R_train_coo = coo_matrix((rating_train,(user_item_train[:,0],user_item_train[:,1])))


	# In[12]:

	print(R_train_coo.data)
	print(R_train_coo.row)
	print(R_train_coo.col)
	print(np.max(R_train_coo.data))
	print(np.max(R_train_coo.row))
	print(np.max(R_train_coo.col))

	# In[14]:

	np.min(R_test_coo.col)


	# In[15]:

	#for training data, we need COO format to calculate training RMSE
	#we need CSR format R when calculate X from \Theta
	#we need CSC format of R when calculating \Theta from X
	assert R_train_coo.nnz == nnz_train
	R_train_coo.row.tofile(test_case_path + 'R_train_coo.row.bin')


	# In[16]:

	R_train_csr = R_train_coo.tocsr()
	R_train_csc = R_train_coo.tocsc()
	R_train_csr.data.astype(dtype).tofile(test_case_path + 'R_train_csr.data.bin')
	R_train_csr.indices.tofile(test_case_path + 'R_train_csr.indices.bin')
	R_train_csr.indptr.tofile(test_case_path + 'R_train_csr.indptr.bin')
	R_train_csc.data.astype(dtype).tofile(test_case_path + 'R_train_csc.data.bin')
	R_train_csc.indices.tofile(test_case_path + 'R_train_csc.indices.bin')
	R_train_csc.indptr.tofile(test_case_path + 'R_train_csc.indptr.bin')


	# In[17]:

	print(R_train_csr.data)
	print(R_train_csr.indptr)
	print(R_train_csr.indices)

	# In[18]:

	print('m = ', m)
	print('n = ', n)
	print('nnz = ', nnz_train)
	print('nnz_test = ', nnz_test)

	# In[19]:
	if u_init is None:
		u_init = np.random.normal(0., 1., [data.shape[0], rank])
	if v_init is None:
		v_init = np.random.normal(0., 1., [data.shape[1], rank])
	u_init.astype(dtype).tofile(test_case_path + 'XT.init.bin')
	v_init.astype(dtype).tofile(test_case_path + 'thetaT.init.bin')

	return u_init, v_init

def recover_test_case(name, rank, dtype = np.float32):

	test_case_path = data_path + name + system_sep
	data_file = test_case_path + 'training_data.npy'
	truth_file = test_case_path + 'ground_truth.npy'

	data = np.load(data_file)
	truth = np.load(truth_file)
	m = data.T.shape[0]
	n = data.T.shape[1]
	u_init = np.fromfile(test_case_path + 'XT.init.bin', dtype = dtype).reshape([m, rank])
	v_init = np.fromfile(test_case_path + 'thetaT.init.bin', dtype = dtype).reshape([n, rank])
	
	return data, truth, u_init, v_init

def read_binary_fp32_file(filename, size):
	return np.fromfile(filename, dtype = np.float32).reshape(size)

def read_binary_fp64_file(filename, size):
	return np.fromfile(filename, dtype = np.float64).reshape(size)

def read_binary_int32_file(filename, size):
	return np.fromfile(filename, dtype = np.int32).reshape(size)

def fake_data():
	a = np.array([[1., 2.], [3., 4.], [5., 6.]])
	b = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
	truth = np.matmul(a, b.T)
	mask = np.array([[1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
	data = np.where(mask == 1, truth, np.nan * truth)
	return data.T, truth.T
def make_fake_data_init(data, truth, rank):
	a, b = make_sparse_files(data, truth, rank)
	return np.ones(a.shape), np.ones(b.shape)

def fake_data_k10():
	a = np.zeros([11, 10])
	b = np.zeros([12, 10])
	for i in range(a.shape[0]):
		a[i, :] = i + 1
	for i in range(b.shape[0]):
		b[i, :] = i + 1
	truth = np.matmul(a, b.T)
	mask = np.zeros(truth.shape)
	from operator import xor
	for r in range(mask.shape[0]):
		for c in range(mask.shape[1]):
			if xor(r % 2 == 0, c % 2 == 0):
				mask[r, c] = 1
	data = np.where(mask == 1, truth, np.nan * truth)
	return data.T, truth.T
def make_fake_data_k10_init(data, truth, rank):
	a = np.ones([data.shape[1], rank])
	b = np.ones([data.shape[0], rank])
	a[:, 0] = 2
	b[:, 0] = 2
	a, b = make_sparse_files(data, truth, rank, a, b)
	return a, b

# original implementation
def collap(vec):
    return "".join(str(x) for x in vec)
    
def outersum1(indvec,y,lam):
    ytemp = y[:,indvec]
    resmat = np.matmul(ytemp,ytemp.transpose())
    resmat = resmat + np.identity(y.shape[0])*lam
    return resmat

def modifiedals(mat, rank, mat1, lam = 3, mu = 1, thres = 0.001, maxit = 20, init_x = None, init_y = None, dtype = np.float64):
	if init_y is None:
		y = np.random.randn(rank,mat.shape[1]).astype(dtype)
	else:
		y = init_y.copy().astype(dtype)
	if init_x is None:
		x = np.empty((rank,mat.shape[0])).astype(dtype)
	else:
		x = init_x.copy().astype(dtype)
	indmat = ~np.isnan(mat)
	
	j = 1
	dist = 1000
	trainerror = 0
	
	start = time.time()
	while(j<maxit and dist>thres):
	    y[:,0] = np.matmul(np.linalg.inv(outersum1(indmat[:,0],x,lam+mu)),(np.matmul(x[:,indmat[:,0]],mat[indmat[:,0],0])+mu*y[:,1]))
	    for i in range(1,y.shape[1]-1):
	        y[:,i] = np.matmul(np.linalg.inv(outersum1(indmat[:,i],x,lam+2*mu)),(np.matmul(x[:,indmat[:,i]],mat[indmat[:,i],i])+mu*(y[:,i-1]+y[:,i+1])))
	    y[:,y.shape[1]-1] = np.matmul(np.linalg.inv(outersum1(indmat[:,y.shape[1]-1],x,lam+mu)),(np.matmul(x[:,indmat[:,y.shape[1]-1]],mat[indmat[:,y.shape[1]-1],y.shape[1]-1])+mu*y[:,y.shape[1]-2]))
	    for i in range(mat.shape[0]):
	        x[:,i] = np.matmul(np.linalg.inv(np.matmul(y[:,indmat[i]],y[:,indmat[i]].transpose())+lam*np.identity(rank)),np.matmul(y[:,indmat[i]],mat[i,indmat[i]]))
	    
	    test = np.matmul(x.transpose(),y)
	    dist = abs(trainerror - np.nansum(abs(test-mat))/sum(sum(indmat)))
	    trainerror = np.nansum(abs(test-mat))/sum(sum(indmat))
	    print('Training Error: ..... ', trainerror)
	    testerror = np.nanmean(abs(test - mat1)[~indmat])
	    print('Testing Error: ...... ',testerror)
	    print('Finished Iteration No. ',j)
	    j = j+1

	end = time.time()
	elapse = end - start
	
	return x, y, trainerror, testerror, elapse

def modifiedals_noinc(mat, rank, mat1, lam = 3, mu = 1, thres = 0.001, maxit = 20, init_x = None, init_y = None, dtype = np.float64):
	if init_y is None:
		y = np.random.randn(rank,mat.shape[1]).astype(dtype)
	else:
		y = init_y.copy().astype(dtype)
	if init_x is None:
		x = np.empty((rank,mat.shape[0])).astype(dtype)
	else:
		x = init_x.copy().astype(dtype)
	indmat = ~np.isnan(mat)
	
	j = 1
	dist = 1000
	trainerror = 0
	y_ = np.zeros(y.shape, dtype = dtype)
	
	start = time.time()
	while(j<maxit and dist>thres):
	    y_[:,0] = np.matmul(np.linalg.inv(outersum1(indmat[:,0],x,lam+mu)),(np.matmul(x[:,indmat[:,0]],mat[indmat[:,0],0])+mu*y[:,1]))
	    for i in range(1,y.shape[1]-1):
	        y_[:,i] = np.matmul(np.linalg.inv(outersum1(indmat[:,i],x,lam+2*mu)),(np.matmul(x[:,indmat[:,i]],mat[indmat[:,i],i])+mu*(y[:,i-1]+y[:,i+1])))
	    y_[:,y.shape[1]-1] = np.matmul(np.linalg.inv(outersum1(indmat[:,y.shape[1]-1],x,lam+mu)),(np.matmul(x[:,indmat[:,y.shape[1]-1]],mat[indmat[:,y.shape[1]-1],y.shape[1]-1])+mu*y[:,y.shape[1]-2]))
	    y = y_
	    for i in range(mat.shape[0]):
	        x[:,i] = np.matmul(np.linalg.inv(np.matmul(y[:,indmat[i]],y[:,indmat[i]].transpose())+lam*np.identity(rank)),np.matmul(y[:,indmat[i]],mat[i,indmat[i]]))
	    
	    test = np.matmul(x.transpose(),y)
	    dist = abs(trainerror - np.nansum(abs(test-mat))/sum(sum(indmat)))
	    trainerror = np.nansum(abs(test-mat))/sum(sum(indmat))
	    print('Training Error: ..... ', trainerror)
	    testerror = np.nanmean(abs(test - mat1)[~indmat])
	    print('Testing Error: ...... ',testerror)
	    print('Finished Iteration No. ',j)
	    j = j+1

	end = time.time()
	elapse = end - start
	
	return x, y, trainerror, testerror, elapse

def test_MAE(estimation, data_mask, truth):
	truth_mask = np.transpose(~np.isnan(truth))
	truth_processed = np.where(np.transpose(truth_mask), truth, np.zeros(truth.shape)).transpose()
	test_mask = np.logical_and(truth_mask, np.logical_not(data_mask))
	test_select = np.where(test_mask, np.ones(test_mask.shape), np.zeros(test_mask.shape))
	N_missing = test_select.sum()
	disc_vector_test = np.multiply(truth_processed - estimation, test_select).flatten()
	return np.linalg.norm(disc_vector_test, 1) / N_missing

def train_MAE(estimation, data_mask, truth):
	truth_mask = np.transpose(~np.isnan(truth))
	truth_processed = np.where(np.transpose(truth_mask), truth, np.zeros(truth.shape)).transpose()
	train_mask = np.logical_and(truth_mask, data_mask)
	train_select = np.where(train_mask, np.ones(train_mask.shape), np.zeros(train_mask.shape))
	N_missing = train_select.sum()
	disc_vector_train = np.multiply(truth_processed - estimation, train_select).flatten()
	return np.linalg.norm(disc_vector_train, 1) / N_missing
