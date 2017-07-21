import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
import time
from config import *


class ALS(object):

	# Parameters
	def __init__(self, rank, Lambda = 0, Mu = 0, MaxIter = 10, tf_Dtype = tf.float32):
		self.rank = rank
		self.tf_Dtype = tf_Dtype
		self.Lambda = Lambda
		self.Mu = Mu
		self.MaxIter = MaxIter;
		self.UpdateParallelIter = 10
		self.LoopParallelIter = 10
		self.log_folder = 'als_log'

	# log path
	log_path = project_path

	# update U
	def UpdateU(self, U_iter, V_iter, data, index_list, index_len):
		"""
		Args:
			U_iter: Tensor
			V_iter: Tensor
			aggregated_data: Tensor (3-by-m-by-n)
			index_len:  Tensor (m-by-2) length of index of each row
		Returns:
			Tensor
		"""
		#left = tf.expand_dims(tf.matmul(data, V_iter, name = 'left_U'), 1)
		def left_row(index_vec):
			return self.__Mu__ * tf.cond(tf.equal(index_vec[0], 0),
						lambda : U_iter[1, :],
						lambda : tf.cond(tf.equal(index_vec[0], U_iter.get_shape()[0] - 1),
							lambda : U_iter[index_vec[0] - 1, :],
							lambda : (U_iter[index_vec[0] + 1, :] + U_iter[index_vec[0] - 1, :])
						)
					)
		ll = tf.expand_dims(tf.matmul(data, V_iter, name = 'left_U'), 1)
		lr = tf.map_fn(left_row, index_len, dtype = self.tf_Dtype, parallel_iterations = self.UpdateParallelIter, name = 'left_smooth')
		left = ll + lr
		def right_row(index_vec):
			V_slice = tf.gather(V_iter, index_list[index_vec[0], 0 : index_vec[1]])
			return tf.matrix_inverse(tf.matmul(V_slice, V_slice, transpose_a = True, name = 'V_x_V') + (self.Lambda +
					tf.cond(tf.logical_or(tf.equal(index_vec[0], 0), tf.equal(index_vec[0], U_iter.get_shape()[0] - 1)), 
						lambda : self.__Mu__,
						lambda : self.__Mu__ * 2.
					)) *  self.__ID_Matrix__, name = 'inv_U')
		right = tf.map_fn(right_row, index_len, dtype = self.tf_Dtype, parallel_iterations = self.UpdateParallelIter, name = 'right_U')

		return tf.squeeze(tf.matmul(left, right))

	# update V
	def UpdateV(self, U_iter, V_iter, data, index_list, index_len):
		"""
		Args:
			U_iter: Tensor
			V_iter: Tensor
			aggregated_data: Tensor (3-by-m-by-n)
			index_len:  Tensor (n-by-2) length of index of each row
		Returns:
			Tensor
		"""
		left = tf.expand_dims(tf.matmul(tf.transpose(data), U_iter, name = 'left_V'), 1)
		def right_row(index_vec):
			U_slice = tf.gather(U_iter, index_list[0 : index_vec[1], index_vec[0]])
			return tf.matrix_inverse(tf.matmul(U_slice, U_slice, transpose_a = True, name = 'U_x_U') + self.__Lambda_ID_Matrix__, name = 'inv_V')
		right = tf.map_fn(right_row, index_len, dtype = self.tf_Dtype, parallel_iterations = self.UpdateParallelIter, name = 'right_V')

		return tf.squeeze(tf.matmul(left, right))

	# single iter
	def ALS_singleiter(self, U_iter, V_iter, data, index_list_U, index_len_U, index_list_V, index_len_V):
		"""
		Args:
			U_iter: Tensor
			V_iter: Tensor
			aggregated_data: Tensor (3-by-m-by-n)
			index_len:  Tensor (m-by-2) length of index of each row
		Returns:
			Tensors U_next, V_next in order
		"""
		U_next = self.UpdateU(U_iter, V_iter, data, index_list_U, index_len_U)
		V_next = self.UpdateV(U_next, V_iter, data, index_list_V, index_len_V)
		return U_next, V_next

	# init before iters
	def __ALS_Init__(self):
		"""
		Args:
		Returns:
		"""
		self.__Lambda__ = tf.constant(self.Lambda, shape = (1, 1), dtype = self.tf_Dtype, name = 'Lambda')
		self.__Mu__ = tf.constant(self.Mu, shape = (1, 1), dtype = self.tf_Dtype, name = 'Mu')
		self.__ID_Matrix__ = tf.eye(self.rank, dtype = self.tf_Dtype, name = 'ID_Matrix')
		self.__Lambda_ID_Matrix__ = self.Lambda * tf.eye(self.rank, dtype = self.tf_Dtype, name = 'Lambda_Matrix')
		self.__c = tf.constant(0, name = 'iter_scaler_stub')

	# main loop
	def ALS_mainloop(self, U_init, V_init, X_data, X_select):
		"""
		Args:
			U_iter: Tensor
			V_iter: Tensor
			X_data: Array
			X_select: Boolean array
		Returns:
			Tensor
		"""
		
		return 
	
	# invoker
	def Run(self, data, data_mask, initU = None, initV = None):
		"""
		Args:
			data: matrix which is going to be imputed, missing data must set to 0
			data_mask: boolean matrix with the same size of data, False for missing data, True otherwise
			initU: initial value of U, if not set, initialize randomly with N(0, 0.1)
			initV: initial value of V, if not set, initialize randomly with N(0, 0.1)
		Returns:
			x, u, v in order
		"""
		self.__M__ = data.shape[0]
		self.__N__ = data.shape[1]
		if initU is None:
			u_shape = [data.shape[0], rank]
			initU = np.random.normal(0, 0.1, u_shape)
		if initV is None:
			v_shape = [data.shape[1], rank]
			initV = np.random.normal(0, 0.1, v_shape)

		index_len_U_		= np.zeros([data.shape[0], 2])
		index_len_V_		= np.zeros([data.shape[1], 2])
		index_list_for_U	= np.zeros(data.shape)
		index_list_for_V	= np.zeros(data.shape)
		for m in range(data.shape[0]):
			list = np.where(data_mask[m, :] == 1)[0]
			index_list_for_U[m, 0 : list.size]	=	list
			index_len_U_[m] = [m, list.size]
		for n in range(data.shape[1]):
			list = np.where(data_mask[:, n] == 1)[0]
			index_list_for_V[0 : list.size, n]	=	list
			index_len_V_[n] = [n, list.size]

		U_init		=	tf.placeholder(self.tf_Dtype,	shape = initU.shape,	name = 'U_init')
		V_init		=	tf.placeholder(self.tf_Dtype,	shape = initV.shape,	name = 'V_init')
		data_		=	tf.placeholder(self.tf_Dtype,	shape = data.shape,		name = 'data')
		index_list_U=	tf.placeholder(tf.int32,		index_list_for_U.shape,	name = 'Index_list_U')
		index_list_V=	tf.placeholder(tf.int32,		index_list_for_V.shape, name = 'Index_list_V')
		index_len_U	=	tf.placeholder(tf.int32,		index_len_U_.shape,		name = 'index_vec_for_U')
		index_len_V	=	tf.placeholder(tf.int32,		index_len_V_.shape,		name = 'index_vec_for_V')

		self.__ALS_Init__()
		als_iter	=	self.ALS_singleiter(U_init, V_init, data_,index_list_U, index_len_U, index_list_V, index_len_V)

		sess		=	tf.Session()
		#options		=	tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
		#run_meta	=	tf.RunMetadata()

		sess.run(tf.global_variables_initializer())
		#merged		=	tf.summary.merge_all()														# tensorboard decoration
		#tmp_logger	=	tf.summary.FileWriter(self.log_path + self.log_folder, graph = sess.graph)		# tensorboard decoration
		
		u, v = initU, initV
		start_time	= time.time()
		for iter in range(self.MaxIter):
			u, v	=	sess.run(als_iter,	{U_init: u, V_init: v, data_: data,
								index_list_U: index_list_for_U, index_list_V: index_list_for_V,
								index_len_U: index_len_U_, index_len_V: index_len_V_})#, options = options, run_metadata = run_meta)
		end_time = time.time()
		print("Compute Time: ", end_time - start_time)
		# Create the Timeline object, and write it to a json file
		#fetched_timeline = timeline.Timeline(run_meta.step_stats)
		#chrome_trace = fetched_timeline.generate_chrome_trace_format()
		#with open('timeline_01.json', 'w') as f:
		#	f.write(chrome_trace)

		x = np.matmul(u, np.transpose(v))

		sess.close()
		#tmp_logger.close()
		return x, u, v

# testing snippet

if __name__ == "__main__":

	# Initialize data
	U_true = np.array([[1., 2.], [3., 4.], [5., 6.]])
	V_true = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
	X_true = np.matmul(U_true, np.transpose(V_true))
	X_mask = np.array([[1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
	X_select = X_mask == 1
	X_data = np.multiply(X_true, X_mask)
	N_observed = X_mask.sum()
	N_missing = X_mask.size - N_observed

	u = np.ones([3, 2])
	v = np.ones([4, 2])

	# initial stats
	print("Init:\n")
	print("U = \n", u)
	print("V = \n", v)
	X_estimate = np.matmul(u, np.transpose(v))
	print("X = \n", X_estimate)
	print("X* = \n", X_true)
	disc = X_estimate - X_true
	print("X - X* = \n", disc)
	print("norm = ", np.linalg.norm(disc, 'fro'))
	disc_vector_train = np.multiply(disc, X_mask).flatten()
	print("train mae = ", np.linalg.norm(disc_vector_train, 1) / N_observed)
	disc_vector_test = np.multiply(disc, 1 - X_mask).flatten()
	print("test mae = ", np.linalg.norm(disc_vector_test, 1) / N_missing)

	# after run
	print('Creating ALS solver...')
	als = ALS(rank = 2, Lambda = 2)
	als.tf_Dtype = tf.float64
	print('Running ALS solver...')
	start_time = time.time()
	X_estimate, u, v = als.Run(X_data, X_mask, u, v)
	end_time = time.time()
	print("Total time: ", end_time - start_time)

	print("iter ", als.MaxIter, ":\n")
	print("U = \n", u)
	print("V = \n", v)
	print("X = \n", X_estimate)
	print("X* = \n", X_true)
	disc = X_estimate - X_true
	print("X - X* = \n", disc)
	disc_vector_train = np.multiply(disc, X_mask).flatten()
	print("train mae = ", np.linalg.norm(disc_vector_train, 1) / N_observed)
	disc_vector_test = np.multiply(disc, 1 - X_mask).flatten()
	print("test mae = ", np.linalg.norm(disc_vector_test, 1) / N_missing)
