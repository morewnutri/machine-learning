#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:35:26 2019

@author: wuhaitao
"""

# Minimize the function
# f(x, y) = (1-x)^2 + 100 * (y-x^2)^2
# f(x, y) = (x^2+y-11)^2 + (x+y^2-7)^2
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import diff, symbols, solve


def func(x, y):  # Rosenbrock
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2  #定义函数


def cal_x_der(a, b):            
    x, y = symbols('x y', real=True)      #将x，y符号化，不会做为变量使用
    f = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2     #函数
    z = diff(f, x)                       #对x的偏导函数赋值给z
    result = z.subs({x: a, y: b})       #在x，y处代入a，b，求得的偏导数赋值给result
    return result


def cal_y_der(a, b):
    x, y = symbols('x y', real=True)
    f = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    z = diff(f, y)
    result = z.subs({x: a, y: b})
    return result


def func_arg_min(pre_x, var, error):     #
    x = pre_x[0] - var * error[0]
    y = pre_x[1] - var * error[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def cal_step_size(pre_x, error, lp, rp, e):     #febonacci method/golden section search
    delta = rp - lp                  #length of the initial interval
    gs = (math.sqrt(5)-1)/2          #斐波那契数列的比率，及两个相邻区间的比率
    p1 = rp - gs * delta             #新区间的y值
    p2 = lp + gs * delta             #新区间的x值
    fx1 = func_arg_min(pre_x, p1, error)
    fx2 = func_arg_min(pre_x, p2, error)
    k = 0
    while abs(rp-lp) > e:
        if fx1 < fx2:
            rp = p2
            p2 = p1
            fx2 = fx1
            det = rp - p2
            if det >= 1e-4:
                p1 = lp + det
            fx1 = func_arg_min(pre_x, p1, error)
        else:
            lp = p1
            p1 = p2
            fx1 = fx2
            det = p1 - lp
            if det >= 1e-4:
                p2 = rp - det
            fx2 = func_arg_min(pre_x, p2, error)
        k = k + 1
    min_point = (p1+p2)/2
    print("step size", min_point)
    return min_point


# def func(x, y):  # Himmelblau
#     return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
#
#
# def cal_x_der(a, b):
#     x, y = symbols('x y', real=True)
#     f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
#     z = diff(f, x)
#     result = z.subs({x: a, y: b})
#     return result
#
#
# def cal_y_der(a, b):
#     x, y = symbols('x y', real=True)
#     f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
#     z = diff(f, y)
#     result = z.subs({x: a, y: b})
#     return result


# def func_arg_min(pre_x, var, temp):
#     x = pre_x[0] + var * temp[0]
#     y = pre_x[1] + var * temp[1]
#     return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def iter_func(max_iter_count=100000):
    pre_x = [0.0, 0.0]            #存储函数的x和y
    pre_xk = [0.0, 0.0]           #存储新点
    error = [0.0, 0.0]            #存储偏导数

    precision = 1e-4              #阈值
    iter_count = 0

    # Rosenbrock plt
    a = np.arange(-2, 2, 0.01)     #从-2到2，以0.01增加，生成一下数字：-2，-2.01，-2.02...也可以用nplinespace（-2，2，100）分100份
    b = np.arange(-2, 5, 0.01)
    [a, b] = np.meshgrid(a, b)     # 生成二维网格Return coordinate matrices from coordinate vectors.
    f = func(a, b)                
    plt.contour(a, b, f, levels=[3, 10, 50, 100, 150], colors='black')     #ab为xy坐标，f为等高线高度，level是
    # # Himmelblau
    # a = np.arange(-5, 5, 0.01)
    # b = np.arange(-5, 5, 0.01)
    # [a, b] = np.meshgrid(a, b)
    # f = func(a, b)
    # plt.contour(a, b, f, levels=[1, 4, 20, 50, 100, 150], colors='black')

    w = np.zeros((10000, 2))    #由0组成的10000维向量，一个数组有两个元素
    w[0, :] = pre_x             #
    while iter_count < max_iter_count:
        f1 = func(pre_x[0], pre_x[1])
        error[0] = round(cal_x_der(pre_x[0], pre_x[1]), 8)     #round用于四舍五入，留下小数点后的八位数
        error[1] = round(cal_y_der(pre_x[0], pre_x[1]), 8)     #error存储函数的偏导数

        step_size = cal_step_size(pre_x, error, 1e-3, 3, 1e-3)

        pre_xk[0] = pre_x[0] - step_size * error[0]        #new point
        pre_xk[1] = pre_x[1] - step_size * error[1]

        f2 = func(pre_xk[0], pre_xk[1])               #此处有改动，求得新点的函数值
        if abs(f2-f1) < precision and math.sqrt((pre_xk[0]-pre_x[0])**2+(pre_xk[1]-pre_x[1])**2) < precision and \
                math.sqrt(cal_x_der(pre_xk[0], pre_xk[1])**2+cal_y_der(pre_xk[0], pre_xk[1])**2) < precision:
            # print("iter_count: ", iter_count, " px ", pre_x[0], " py ", pre_x[1])
            break
        else:
            pre_x[0] = round(pre_xk[0], 8)
            pre_x[1] = round(pre_xk[1], 8)
            iter_count += 1
            w[iter_count, :] = pre_x
            print("iter_count: ", iter_count, " px ", pre_x[0], " py ", pre_x[1])
    plt.plot(w[:, 0], w[:, 1], 'r+', w[:, 0], w[:, 1])
    plt.show()
    return pre_x


if __name__ == '__main__':
    iter_func()