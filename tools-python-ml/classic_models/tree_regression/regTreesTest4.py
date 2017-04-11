# -*- coding: utf-8 -*-
# !/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import cart_reg_tree

myMat = cart_reg_tree.loadDataSet('ex2.txt')
myMat = np.mat(myMat)

tree = cart_reg_tree.createTree(myMat)
print("tree", tree)

plt.plot(myMat[:, 0], myMat[:, 1], 'ro')
plt.show()
