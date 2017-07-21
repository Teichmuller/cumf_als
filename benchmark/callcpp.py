import subprocess
from config import *

name = 'test_case3_1M'
subprocess.call([base_path + 'Release/gpu-accelerated-analytics', '96', '1000000', '20', '17171442', '22426274', '3.0', '2.0', '1', '1', '1', data_path + name])

