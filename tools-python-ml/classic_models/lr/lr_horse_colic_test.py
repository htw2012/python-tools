#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= "马儿疝气病测试"
author= "huangtw"
ctime = 2017/04/10
"""
from lr import *

def load_horse_colic(filename):
    """
    数据载入
    :param filename:
    :return:
    """
    frTrain = open(filename)
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        lineArr.append(1.0)  # 开始加一个1,用于学习偏置bias
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    return mat(trainingSet), mat(trainingLabels).transpose()


# step 1: load data
print "step 1: load data..."
train_x, train_y = load_horse_colic('data/horseColicTraining.txt')
test_x, test_y = load_horse_colic('data/horseColicTest.txt')
print("train_x-shape", train_x.shape, "train_y-shape", train_y.shape)

## step 2: training...
print "step 2: training..."
# opts = {'alpha': 0.01, 'maxIter': 10000, 'optimizeType': 'stocGradDescent'}#(close test) is: 67.893%,(open test) is: 70.149%
opts = {'alpha': 0.01, 'maxIter': 30000,
        'optimizeType': 'gradDescent'}  # (close test) is: 67.559%,(open test) is: 74.627%
# opts = {'alpha': 0.01, 'maxIter': 5000, 'optimizeType': 'smoothStocGradDescent'}#(close test) is: 62.876%,(open test) is: 68.657%

optimalWeights = fit(train_x, train_y, opts)

## step 3: testing
print "step 3: testing..."
accuracy = evaluate(optimalWeights, train_x, train_y)
print 'The classify accuracy(close test) is: %.3f%%' % (accuracy * 100)

accuracy = evaluate(optimalWeights, test_x, test_y)
print 'The classify accuracy(open test) is: %.3f%%' % (accuracy * 100)
