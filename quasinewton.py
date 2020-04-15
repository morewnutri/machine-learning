#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 10:32:22 2019

@author: wuhaitao
"""

import random
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

def func(x, y):#目标函数
    result = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    # result = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
    return result

def get_dict(res_dict):#获取坐标：值的dict
    for i in range(3):
        x = random.randint(-10, 10)       #范围之内随机生成生成三个数
        y = random.randint(-10, 10)
        result = func(x, y)
        loc = (x, y)                 #loc是存储坐标的元组
        res_dict[loc] = result      #res_dict 存储的是点的坐标对应的函数值
    return res_dict

def get_max(res_dict):#获取最大函数值在dict中的index
    flag = 0
    value_list = res_dict.values()   
    for value in value_list:
        max_value = value
        break
    for r in res_dict:
        if res_dict[r] >= max_value:
            max_value = res_dict[r]
            index = flag
            flag += 1
        else:
            flag += 1
    return index

def get_min(res_dict):#获取最小函数值在dict中的index
    flag = 0
    value_list = res_dict.values()
    for value in value_list:
        min_value = value
        break
    for r in res_dict:
        if res_dict[r] <= min_value:
            min_value = res_dict[r]
            index = flag
            flag += 1
        else:
            flag += 1
    return index

def get_mid(res_dict):#获取中间坐标）
    x = 0
    y = 0
    for res in res_dict:
        x += res[0]
        y += res[1]
    mean_x = x/len(res_dict)
    mean_y = y/len(res_dict)
    mean = [mean_x, mean_y]
    return mean

def get_z1(index, mean, res_dict):
    flag = 0
    for res in res_dict:
        if flag == index:
            cord = res
            break
        else:
            flag += 1
        #cord是（x,y）
    new_cord_x = mean[0] - (cord[0] - mean[0])
    new_cord_y = mean[1] - (cord[1] - mean[1])
    cord_z1 = [new_cord_x, new_cord_y]
    z1 = func(new_cord_x, new_cord_y)
    result = [cord_z1, z1]
    return result

def get_z2(index, mean, res_dict):
    flag = 0
    for res in res_dict:
        if flag == index:
            cord = res
            break
        else:
            flag += 1
    new_cord_x = mean[0] - 2*(cord[0] - mean[0])
    new_cord_y = mean[1] - 2*(cord[1] - mean[1])
    cord_z2 = [new_cord_x, new_cord_y]
    z2 = func(new_cord_x, new_cord_y)
    result = [cord_z2, z2]
    return result

def get_z3(index, mean, res_dict):
    flag = 0
    for res in res_dict:
        if flag == index:
            cord = res
            break
        else:
            flag += 1
    new_cord_x = cord[0] - 0.5*(cord[0] - mean[0])
    new_cord_y = cord[1] - 0.5*(cord[1] - mean[1])
    cord_z3 = [new_cord_x, new_cord_y]
    z3 = func(new_cord_x, new_cord_y)
    result = [cord_z3, z3]
    return result

def get_z4(index_min, index_max, mean, res_dict):
    flag = 0
    for res in res_dict:
        if flag == index_min:
            cord_min = res
            break
        else:
            flag += 1
    flag = 0
    for res in res_dict:
        if flag == index_max:
            cord_max = res
            break
        else:
            flag += 1
    new_cord_x = cord_max[0] - 0.5*(cord_max[0] - cord_min[0])
    new_cord_y = cord_max[1] - 0.5*(cord_max[1] - cord_min[1])
    cord_z4 = [new_cord_x, new_cord_y]
    z4 = func(new_cord_x, new_cord_y)
    result = [cord_z4, z4]
    return result
simple_list = []
if __name__ == '__main__':
    simple = []
    x_list = []
    y_list = []
    res_dict = {}
    res_dict = get_dict(res_dict)#获取随机生成的四个坐标和计算所得的函数值
    for key in res_dict:
        simple.append(key)     #用于存储字典里的键
    simple_list.append(simple)
    tol = 0.0000001
    loss_list = []
    iter_list = []
    iter = 1
    while(True):
        simple = []
        index_max = get_max(res_dict) # 求出其中的最大值
        index_min = get_min(res_dict) # 求出其中的最小值
        flag_res = 0
        tag = 0
        for res in res_dict:#获取index对应的value函数值
            if flag_res == index_max:
                max_value = res_dict[res]
                tag += 1
                flag_res += 1
            elif flag_res == index_min:
                min_value = res_dict[res] # 求出最小的值
                tag += 1
                flag_res += 1
            else:
                if tag == 2:
                    break
                else:
                    flag_res += 1
        if abs(max_value - min_value) <= tol:#满足最优解条件时
            print(res_dict)
            break
        else:
            mean = get_mid(res_dict)
            result_list = []#分析四种分散方法对应的z函数值
            result_list.append(get_z1(index_max, mean, res_dict))
            result_list.append(get_z2(index_max, mean, res_dict))
            result_list.append(get_z3(index_max, mean, res_dict))
            result_list.append(get_z4(index_max, index_min, mean, res_dict))
            index = 0
            flag = 0
            min = result_list[0][1]#得到最小的函数值
            for r in result_list:#获取最小的z值对应的坐标进行替换
                if result_list[flag][1] <= min:
                    min = result_list[index][1]
                    index = flag
                    flag += 1
                else:
                    flag += 1
            x_list.append(result_list[0][0][0])
            y_list.append(result_list[0][0][1])
            z = result_list[index][1]
            z_cord = tuple(result_list[index][0])#(x,y)
            tmp = 0
            for res in res_dict:#去掉最大的点
                if tmp == index_max:
                    res_dict.pop(res)
                    break
                else:
                    tmp += 1
            res_dict[z_cord] = z#更新dict
            for key in res_dict:
                simple.append(key)
            simple_list.append(simple)
            print(z)

        iter_list.append(iter)
        loss_list.append(abs(max_value - min_value))
        iter += 1





import os
count = 0
for simplex in simple_list[:15]:
    plt.cla()
    n = 1000
    # x1 = np.linspace(-20, 20, n)
    # x2 = np.linspace(-20, 20, n)
    # X, Y = np.meshgrid(x1, x2)
    # Z = np.sqrt(X ** 2 + Y ** 2)
    # plt.contour(X, Y, Z, levels=list(np.arange(0, 40, 0.5)))
    plt.gca().set_aspect("equal")
    plt.xlim((-20, 20))
    plt.ylim((-20, 20))
    plt.plot([simplex[0][0], simplex[1][0]],
             [simplex[0][1], simplex[1][1]], color="#000000")
    plt.plot([simplex[1][0], simplex[2][0]],
             [simplex[1][1], simplex[2][1]], color="#000000")
    plt.plot([simplex[2][0], simplex[0][0]],
             [simplex[2][1], simplex[0][1]], color="#000000")
    plt.savefig("{}.png".format(count))
    count += 1
    plt.show()
