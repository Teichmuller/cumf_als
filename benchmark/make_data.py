import numpy as np
from config import *
from utils import make_test_case

data_file = exchange_data_folder + 'excavated_data.npy'
truth_file = exchange_data_folder + 'data.npy'
rank = 20

data, truth = (np.load(data_file)[0 : 1000000, :], np.load(truth_file)[0 : 1000000, :])
#init_y, init_x = make_fake_data_k10_init(data, truth, rank)
init_y, init_x = make_test_case('test_case3_1M', data, truth, rank)
#data, truth = fake_data_k10()
