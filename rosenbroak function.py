#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:22:51 2019

@author: wuhaitao
"""

import numpy as np


def cal_rosenbrock(x, y):
    """
    计算rosenbrock函数的值
    :param x:
    :param y:
    :return:
    """
    return (1 - x) ** 2 + 100 * (y - x** 2) ** 2


def cal_rosenbrock_prax(x, y):
    """
    对x求偏导
    """
    return -2 + 2 * x - 400 * (y - x ** 2) * x

def cal_rosenbrock_pray(x, y):
    """
    对y求偏导
    """
    return 200 * (y - x ** 2)

def for_rosenbrock_func(max_iter_count=100000, step_size=0.001):
    pre_x = np.zeros((2,), dtype=np.float32)  #创建prex多维数组，2行多列浮点型
    loss = 10
    iter_count = 0
    while loss > 0.001 and iter_count < max_iter_count:
        error = np.zeros((2,), dtype=np.float32)
        error[0] = cal_rosenbrock_prax(pre_x[0], pre_x[1])#error数组用于存储偏导数
        error[1] = cal_rosenbrock_pray(pre_x[0], pre_x[1])

        for j in range(2):
            pre_x[j] -= step_size * error[j]  #起始点为0，0.每次循环从一个点到另个点
                       #因为error每次都在变所以可以认为step.zise*erroe就是步长且每次在变
        loss = cal_rosenbrock(pre_x[0], pre_x[1])  # 最小值为0，loss存储的是函数值，直到最小值0

        print("iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    return pre_x

if __name__ == '__main__':
    w = for_rosenbrock_func()  
    print(w)
