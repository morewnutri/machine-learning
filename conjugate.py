import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import ticker

N = 1

x_list = []
y_list = []
def func(x, y):#目标函数
    result = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    # result = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
    return result

f = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2  # 第一条公式
# f = lambda x: (x[0]**2 + x[1] - 11) ** 2 + (x[0] + x[1]**2 - 7) ** 2 # 第二条公式

cost = lambda x: np.sum(f(x))


alpha = lambda g1, g0: g1.T.dot(g1) / g0.T.dot(g0)


def grad(x):
    gx0 = -2 + 2 * x[0] - 400 * (x[1] - x[0] ** 2) * x[0] # 第一条公式
    # gx0 = 2 * (x[0]**2 + x[1] - 11) * 2 * x[0] + 2 * (x[0] + x[1] ** 2 - 7) # 第二条公式
    gx1 = 200 * (x[1] - x[0] ** 2) # 第一条公式
    # gx1 = 2 * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7) * 2 * x[1] # 第二条公式
    return np.array([np.sum(gx0), np.sum(gx1)])





def gd_alg(x_init, epsilon=1e-1, m_lambda=0.0001):
    current_cost = 10
    start_time = time.time()
    xt = x_init
    costs = [cost(xt)]
    gt = grad(xt)
    loss_list = []
    iter_list = []
    iter = 1
    while current_cost > 0.000001:
        xt1 = xt.T - m_lambda*gt
        gt1 = grad(xt1.T)

        current_cost = cost(xt1.T)
        costs.append(current_cost)

        xt = xt1.T
        x_list.append(xt[0])
        y_list.append(xt[1])
        gt = gt1
        print(current_cost)
        iter_list.append(iter)
        loss_list.append(current_cost)
        iter += 1
    print("GD: xt^* =\n", xt)
    print("GD: time usage =", time.time() - start_time)
    print("GD: iter_times =", len(costs))
    return costs, xt, iter_list, loss_list



if __name__ == '__main__':
    np.random.seed(121)
    X0 = np.random.randn(2, )
    costs_gd, xt_gd, iter_list, loss_list = gd_alg(x_init=X0)
    n = 256
    # 定义x, y
    x = np.linspace(-1, 1.1, n)  # x，y的起始点为-1，-1。1.1是坐标轴的长度
    y = np.linspace(-1, 1.1, n)

    # 生成网格数据
    X, Y = np.meshgrid(x, y)

    plt.figure()
    # 填充等高线的颜色, 8是等高线分为几部分
    plt.contourf(X, Y, func(X, Y), 5, alpha=0, cmap=plt.cm.hot)
    # 绘制等高线
    C = plt.contour(X, Y, func(X, Y), 8, locator=ticker.LogLocator(), colors='black', linewidth=0.01)
    # 绘制等高线数据
    plt.clabel(C, inline=True, fontsize=10)
    # ---------------------

    # def H(x, y):
    #     return np.matrix([[1200 * x * x - 400 * y + 2, -400 * x],  # 由二阶偏导系数构成的海森矩阵
    #                       [-400 * x, 200]])
    #
    #
    # def delta_newton(x, y):
    #     alpha = 1.0
    #     delta = alpha * H(x, y).I * grad_1(x, y)  # 海森矩阵的逆矩阵
    #     return delta
    #
    #
    # x = np.matrix([[-0.5],  # 起点
    #                [0.5]])
    #
    # tol = 0.00001
    # xv = [x[0, 0]]  # 不知道是啥
    # yv = [x[1, 0]]

    plt.plot(x_list[0], y_list[0], marker='o')  # 不知道是啥
    #
    # for t in range(100):
    #     delta = delta_newton(x[0, 0], x[1, 0])
    #     if abs(delta[0, 0]) < tol and abs(delta[1, 0]) < tol:
    #         break
    #     x = x - delta  # move to next point
    #     xv.append(x[0, 0])  # 数组拼接
    #     yv.append(x[1, 0])

    plt.plot(x_list, y_list, label='track')
    # plt.plot(xv, yv, label='track', marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('conjudate Method for Rosenbrock Function')
    plt.legend()
    plt.show()



