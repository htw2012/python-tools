#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= "马儿疝气病测试"
author= "huangtw"
ctime = 2017/04/10
"""
from lr import *
## step 1: load data
print "step 1: load data..."


def load_horse_colic():
    frTrain = open('horseColicTraining.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    frTest = open('horseColicTest.txt')
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
    return frTest


train_x, train_y = load_horse_colic()
test_x = train_x
test_y = train_y

## step 2: training...
print "step 2: training..."
# opts = {'alpha': 0.01, 'maxIter': 2000, 'optimizeType': 'stocGradDescent'}# 95.000%
# opts = {'alpha': 0.01, 'maxIter': 2000, 'optimizeType': 'gradDescent'} # 95.000%
opts = {'alpha': 0.01, 'maxIter': 30, 'optimizeType': 'smoothStocGradDescent'}# 96.000%

optimalWeights = fit(train_x, train_y, opts)

## step 3: testing
print "step 3: testing..."
accuracy = evaluate(optimalWeights, test_x, test_y)

## step 4: show the result
print "step 4: show the result..."
print 'The classify accuracy(close test) is: %.3f%%' % (accuracy * 100)
