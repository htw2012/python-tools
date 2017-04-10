#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= "马儿疝气病测试"
author= "huangtw"
ctime = 2017/04/10
"""
from lr import *


def load_horse_colic(filename):
    frTrain = open(filename)
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')

        lineArr = []
        lineArr.append(1.0)  # 开始加一个偏置
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
        return mat(trainingSet), mat(trainingLabels).transpose()

## step 1: load data
print "step 1: load data..."
train_x, train_y = load_horse_colic('data/horseColicTraining.txt')

test_x, test_y = load_horse_colic('data/horseColicTest.txt')

## step 2: training...
print "step 2: training..."
# opts = {'alpha': 0.01, 'maxIter': 2000, 'optimizeType': 'stocGradDescent'}#
opts = {'alpha': 0.01, 'maxIter': 2, 'optimizeType': 'gradDescent'}  #
# opts = {'alpha': 0.01, 'maxIter': 30, 'optimizeType': 'smoothStocGradDescent'}#

optimalWeights = fit(train_x, train_y, opts)

## step 3: testing
print "step 3: testing..."
accuracy = evaluate(optimalWeights, train_x, train_y)
print 'The classify accuracy(close test) is: %.3f%%' % (accuracy * 100)

accuracy = evaluate(optimalWeights, test_x, test_y)
print 'The classify accuracy(open test) is: %.3f%%' % (accuracy * 100)
