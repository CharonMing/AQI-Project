import os
import numpy as np


class dataset():
    """dataset"""

    def __init__(self, root):
        self.root = root
        self.load_data()

    def load_data(self):
        f = open(self.root, 'r', encoding='UTF-8')
        lines = f.readlines()
        rows = len(lines)
        datamat = np.zeros((rows - 1, 7))
        row = 0
        for line in lines:
            line = line.strip().split('\t')
            if row == 0:
                row += 1
            else:
                datamat[row - 1, :] = line[1:]
                row += 1
        return datamat
