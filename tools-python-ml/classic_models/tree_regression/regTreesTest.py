# -*- coding: utf-8 -*-
# !/usr/bin/env python

import numpy as np

import cart_reg_tree

testMat = np.mat(np.eye(4))
print("testMat", testMat)

# idx=1的特征进行切分，即第二个特征进行切分
feature = 1
value = 0.5

# 针对第一个特征进行分割，如果大于0.5为一组，小于0.5为另一组
mat0, mat1 = cart_reg_tree.binSplitDataSet(testMat, feature, value)

print("mat0", mat0)
print("mat1", mat1)
