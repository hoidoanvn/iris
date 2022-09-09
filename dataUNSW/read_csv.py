import numpy as np
import pandas as pd

# để None bằng số mấy thì nó lấy dòng đó làm header
path = r'D:\condamini\MachineLearningTutorail\Tutorial1\dataUNSW\UNSW_NB15_testing-set.csv'
file_test = pd.read_csv(path, header=None)

# file_test[3] # in ra cột số 3
print(file_test[2])
