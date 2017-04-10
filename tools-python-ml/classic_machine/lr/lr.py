#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= "
lr模型，支持三种训练方法：
1.

"
author= "huangtw"
ctime = 2017/04/10
"""
from numpy import *
import matplotlib.pyplot as plt
import time


# calculate the sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def fit(train_x, train_y, opts):
    '''
    模型训练
    :param train_x:
    :param train_y:
    :param opts:
    :return:
    '''
    # calculate training time
    startTime = time.time()

    numSamples, numFeatures = shape(train_x)
    alpha = opts['alpha']
    maxIter = opts['maxIter']
    print("opts", opts)
    weights = ones((numFeatures, 1)) # 权值为n*1

    # optimize through gradient descent algorilthm
    for k in range(maxIter):
        if opts['optimizeType'] == 'gradDescent': # 单纯的梯度下降法
            output = sigmoid(train_x * weights)
            error = train_y - output
            weights = weights + alpha * train_x.transpose() * error

        elif opts['optimizeType'] == 'stocGradDescent': # 随机梯度下降法
            for i in range(numSamples):
                output = sigmoid(train_x[i, :] * weights)
                error = train_y[i, 0] - output
                weights = weights + alpha * train_x[i, :].transpose() * error

        elif opts['optimizeType'] == 'smoothStocGradDescent': # smooth SGD
            # randomly select samples to optimize for reducing cycle fluctuations
            dataIndex = range(numSamples)
            for i in range(numSamples):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int(random.uniform(0, len(dataIndex)))
                output = sigmoid(train_x[randIndex, :] * weights)
                error = train_y[randIndex, 0] - output
                weights = weights + alpha * train_x[randIndex, :].transpose() * error
                del(dataIndex[randIndex]) # during one interation, delete the optimized sample
        else:
            raise NameError('Not support optimize method type!')


    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    return weights


def evaluate(weights, test_x, test_y):
    '''
    模型评估
    :param weights:
    :param test_x:
    :param test_y:
    :return:
    '''
    numSamples, numFeatures = shape(test_x)
    matchCount = 0
    for i in xrange(numSamples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy
