import os
import numpy as np
# from sklearn.model_selection import train_test_split
# class dataset():
# 	"""dataset"""
# 	def __init__(self, root, test_size, random_state=0,years=None):
# 		super(dataset, self).__init__()
# 		self.root = root
# 		self.test_size = test_size
# 		self.random_state = random_state
# 		self.years = years
	
def load_data():
	f = open("../data/data.txt")
	lines = f.readlines()
	rows = len(lines)
	datamat = np.zeros((rows-1, 7))
	row = 0
	for line in lines:
		line = line.strip().split('\t')
		if row==0:
			row+=1
		else:
			print(row,line[1:])
			datamat[row-1,:] = line[1:]
			row+=1
	return datamat

print(load_data().shape)

    # def split_data():