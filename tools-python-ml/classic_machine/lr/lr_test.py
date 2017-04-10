#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= ""
author= "huangtw"
ctime = 2017/04/10
"""
from lr import *
from numpy import *

# show your trained logistic regression model only available with 2-D data
def showLogRegres(weights, train_x, train_y):
    # notice: train_x and train_y is mat datatype
    numSamples, numFeatures = shape(train_x)
    if numFeatures != 3:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    # draw all samples
    for i in xrange(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

    # draw the classify line
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    weights = weights.getA()  # convert mat to array
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def loadData():
    train_x = []
    train_y = []
    fileIn = open('data/testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return mat(train_x), mat(train_y).transpose()


## step 1: load data
print "step 1: load data..."
train_x, train_y = loadData()
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
showLogRegres(optimalWeights, train_x, train_y)