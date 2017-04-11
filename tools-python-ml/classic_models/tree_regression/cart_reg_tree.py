#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= ""
author= "huangtw"
ctime = 2017/04/11
"""
import numpy as np


# 载入数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # fltLine = list(map(float, curLine))# python3
        fltLine = map(float, curLine)  # python2
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):  # 数据集
    '''
    切分数据集为两个子集
    :param dataSet:
    :param feature: 待切分特征
    :param value: 特征值
    :return:
    '''
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    # 下面原书代码报错 index 0 is out of bounds,使用上面两行代码
    # mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    # mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0, mat1


# Tree结点类型：回归树
def regLeaf(dataSet):  # 生成叶结点，在回归树中是目标变量特征的均值
    return np.mean(dataSet[:, -1])


# 误差计算函数：回归误差
def regErr(dataSet):  # 计算目标的平方误差（均方误差*总样本数）
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


# 二元切分
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''
    二元切分
    :param dataSet:
    :param leafType:
    :param errType:
    :param ops:
    :return:
    '''
    # 切分特征的参数阈值，用户初始设置好
    tolS = ops[0]  # 允许的误差下降值
    tolN = ops[1]  # 切分的最小样本数
    # 若所有特征值都相同，停止切分
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # 倒数第一列转化成list 不重复
        return None, leafType(dataSet)  # 如果剩余特征数为1，停止切分1。
        # 找不到好的切分特征，调用regLeaf直接生成叶结点
    m, n = np.shape(dataSet)
    S = errType(dataSet)  # 最好的特征通过计算平均误差
    bestS = np.inf
    bestIndex = 0
    bestValue = 0

    for featIndex in range(n - 1):  # 遍历数据的每个属性特征
        # for splitVal in set(dataSet[:,featIndex]): python3报错修改为下面
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):  # 遍历每个特征里不同的特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)  # 对每个特征进行二元分类
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue

            newS = errType(mat0) + errType(mat1)
            if newS < bestS:  # 更新为误差最小的特征
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果切分后误差效果下降不大，则取消切分，直接创建叶结点
    if (S - bestS) < tolS:
        return None, leafType(dataSet)  # 停止切分2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)

    # 判断切分后子集大小，小于最小允许样本数停止切分3
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue  # 返回特征编号和用于切分的特征值


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''
    创建树
    :param dataSet:
    :param leafType:
    :param errType:
    :param ops:
    :return:
    '''
    # 数据集默认NumPy Mat 其他可选参数【结点类型：回归树，误差计算函数，ops包含树构建所需的其他元组】
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val  # 满足停止条件时返回叶结点值

    # 切分后赋值
    retTree = {}
    retTree['spInd'] = feat  # 选择的特征作为划分点
    retTree['spVal'] = val  # 特征划分点的值
    # 切分后的左右子树
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def prune(tree, testData):
    '''
    根据测试数据进行剪枝的操作
    :param tree:
    :param testData:
    :return:
    '''
    if np.shape(testData)[0] == 0:
        return getMean(tree)  # if we have no test data collapse the tree

    if (isTree(tree['right']) or isTree(tree['left'])):  # if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)

    # if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])

    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0
