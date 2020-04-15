#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 10:48:28 2019

@author: wuhaitao
"""


# Minimize the function
# f(x, y) = (1-x)^2 + 100 * (y-x^2)^2
# f(x, y) = (x^2+y-11)^2 + (x+y^2-7)^2
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, symbols


# def func(x, y):  # Rosenbrock
#     return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
#
#
# def cal_x_der(a, b):
#     x, y = symbols('x y', real=True)
#     f = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
#     z = diff(f, x)
#     result = z.subs({x: a, y: b})
#     return result
#
#
# def cal_y_der(a, b):
#     x, y = symbols('x y', real=True)
#     f = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
#     z = diff(f, y)
#     result = z.subs({x: a, y: b})
#     return result
#
#
# def func_arg_min(pre_x, var, temp):
#     x = pre_x[0] + var * temp[0]
#     y = pre_x[1] + var * temp[1]
#     return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
#
#
# def cal_step_size(pre_x, temp, lp, rp, e):
#     delta = rp - lp
#     gs = (math.sqrt(5)-1)/2
#     p1 = rp - gs * delta
#     p2 = lp + gs * delta
#     fx1 = func_arg_min(pre_x, p1, temp)
#     fx2 = func_arg_min(pre_x, p2, temp)
#     k = 0
#     while abs(rp-lp) > e:
#         if fx1 < fx2:
#             rp = p2
#             p2 = p1
#             fx2 = fx1
#             det = rp - p2
#             if det >= 1e-4:
#                 p1 = lp + det
#             fx1 = func_arg_min(pre_x, p1, temp)
#         else:
#             lp = p1
#             p1 = p2
#             fx1 = fx2
#             det = p1 - lp
#             if det >= 1e-4:
#                 p2 = rp - det
#             fx2 = func_arg_min(pre_x, p2, temp)
#         k = k + 1
#     min_point = (p1+p2)/2
#     # print("step size", min_point)
#     return min_point


def func(x, y):  # Himmelblau
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def cal_x_der(a, b):
    x, y = symbols('x y', real=True)
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    z = diff(f, x)
    result = z.subs({x: a, y: b})
    return result


def cal_y_der(a, b):
    x, y = symbols('x y', real=True)
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    z = diff(f, y)
    result = z.subs({x: a, y: b})
    return result


def func_arg_min(pre_x, var, temp):     #
    x = pre_x[0] + var * temp[0]
    y = pre_x[1] + var * temp[1]
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def cal_step_size(pre_x, temp, lp, rp, e):
    delta = rp - lp
    gs = (math.sqrt(5)-1)/2
    p1 = rp - gs * delta
    p2 = lp + gs * delta
    fx1 = func_arg_min(pre_x, p1, temp)
    fx2 = func_arg_min(pre_x, p2, temp)
    k = 0
    while abs(rp-lp) > e:
        if fx1 < fx2:
            rp = p2
            p2 = p1
            fx2 = fx1
            det = rp - p2
            if det >= 1e-4:
                p1 = lp + det
            fx1 = func_arg_min(pre_x, p1, temp)
        else:
            lp = p1
            p1 = p2
            fx1 = fx2
            det = p1 - lp
            if det >= 1e-4:
                p2 = rp - det
            fx2 = func_arg_min(pre_x, p2, temp)
        k = k + 1
    min_point = (p1+p2)/2
    # print("step size", min_point)
    return min_point


def iter_func(max_iter_count=100000):
    pre_x = [-1.0, 0.0]                      #起点
    pre_xk = [0.0, 0.0]
    error1 = np.mat([0.0, 0.0]).reshape(-1, 1)  #mat create matrix,reshape表示二阶矩阵，-1表示自动计算，error1存储偏导数
    error2 = np.mat([0.0, 0.0]).reshape(-1, 1) #存储求得的新的点的偏导数
    s = np.mat([0.0, 0.0]).reshape(-1, 1)     #s存储求得两个点的差值

    h = np.diag(np.array([1.0, 1.0]))         #diagarray是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵 array是一个二维矩阵时，结果输出矩阵的对角线元素
    i = np.diag(np.array([1.0, 1.0]))         

    precision = 0.001
    iter_count = 0

    # # Rosenbrock plt
    # a = np.arange(-2, 2, 0.01)
    # b = np.arange(-2, 5, 0.01)
    # [a, b] = np.meshgrid(a, b)
    # f = func(a, b)
    # plt.contour(a, b, f, levels=[3, 10, 50, 100, 150], colors='black')
    # Himmelblau
    a = np.arange(-5, 5, 0.01)         #在给定的时间间隔内返回均匀间隔的值。0.01 为步长
    b = np.arange(-5, 5, 0.01)
    [a, b] = np.meshgrid(a, b)          #生成二维网格Return coordinate matrices from coordinate vectors.
    f = func(a, b)
    plt.contour(a, b, f, levels=[1, 4, 20, 50, 100, 150], colors='black')

    w = np.zeros((10000, 2))        #由0组成的10000维向量，一个数组有两个元素
    w[0, :] = pre_x                 #
    while iter_count < max_iter_count:
        f1 = func(pre_x[0], pre_x[1])

        error1[0] = cal_x_der(pre_x[0], pre_x[1])
        error1[1] = cal_y_der(pre_x[0], pre_x[1])
        temp = - h * error1         #将x偏导数和y偏导数存储在temp矩阵，按照课件，这一步是p0。

        step_size = cal_step_size(pre_x, temp, 1e-3, 1.0, 1e-3)

        pre_xk[0] = pre_x[0] + step_size * temp[0]  #寻找下一个点
        pre_xk[1] = pre_x[1] + step_size * temp[1]

        f2 = func(pre_xk[0], pre_xk[1])      #求在这个点的函数值
        if abs(f2-f1) < precision and math.sqrt((pre_xk[0]-pre_x[0])**2+(pre_xk[1]-pre_x[1])**2) < precision and \
                math.sqrt(cal_x_der(pre_xk[0], pre_xk[1]) ** 2 + cal_y_der(pre_xk[0], pre_xk[1]) ** 2) < precision:
            # print("iter_count: ", iter_count, " px ", pre_x[0], " py ", pre_x[1])
            break
        else:
            error2[0] = cal_x_der(pre_xk[0], pre_xk[1])
            error2[1] = cal_y_der(pre_xk[0], pre_xk[1])
            y = error2 - error1       #偏导数的差，y是数组

            s[0] = pre_xk[0] - pre_x[0]
            s[1] = pre_xk[1] - pre_x[1]

            p = float(1.0 / (np.transpose(y) * s))    #transpose改变相应序列，转置
            part1 = i - p * s * np.transpose(y)     
            part2 = i - p * y * np.transpose(s)
            part3 = p * s * np.transpose(s)

            h = part1 * h * part2 + part3

            pre_x[0] = pre_xk[0]
            pre_x[1] = pre_xk[1]
            iter_count += 1
            w[iter_count, :] = pre_x
            print("iter_count: ", iter_count, " px ", pre_x[0], " py ", pre_x[1])
    plt.plot(w[:, 0], w[:, 1], 'r+', w[:, 0], w[:, 1])
    plt.show()
    return pre_x


if __name__ == '__main__':
    iter_func()