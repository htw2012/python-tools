#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= "
lr模型，支持三种训练方法：
1.gd
2.sgd
3.smooth-sgd

梯度下降：
grda
"
author= "huangtw"
ctime = 2017/04/10
"""
import time

from numpy import *

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

    # 梯度下降的基本原理是delta_theta=-eta*grad_E
    for k in range(maxIter):

        if opts['optimizeType'] == 'gradDescent': # 单纯的梯度下降法
            output = sigmoid(train_x * weights)
            grad_weight = train_x.transpose() * (output - train_y)  # 梯度部分
            delta_weight = - alpha * grad_weight
            weights += delta_weight  # 梯度下降

        elif opts['optimizeType'] == 'stocGradDescent': # 随机梯度下降法
            for i in range(numSamples):  # 一次训练完才算一个epoch
                output = sigmoid(train_x[i, :] * weights)
                grad_weight = train_x[i, :].transpose() * (output - train_y[i, 0])
                delta_weight = - alpha * grad_weight
                weights += delta_weight

        elif opts['optimizeType'] == 'smoothStocGradDescent': # smooth SGD
            # randomly select samples to optimize for reducing cycle fluctuations
            dataIndex = range(numSamples)
            for i in range(numSamples):
                randIndex = int(random.uniform(0, len(dataIndex)))
                output = sigmoid(train_x[randIndex, :] * weights)
                grad_weight = (output - train_y[randIndex, 0]) * train_x[randIndex, :].transpose()

                alpha = 4.0 / (1.0 + k + i) + 0.01
                delta_weight = - alpha * grad_weight
                weights += delta_weight

                del(dataIndex[randIndex]) # during one interation, delete the optimized sample
        else:
            raise NameError('Not support optimize method type!')

    print 'training complete! Took %fs!' % (time.time() - startTime)
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
