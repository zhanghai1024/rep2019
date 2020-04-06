import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def crra_utility(x, alfa):
    y = (1 - np.power(x, -alfa)) / alfa

    return y


def plot_utility_fun():
    x = np.arange(0.2, 3, 0.01)
    y = crra_utility(x, -1)
    plt.figure(num=None, figsize=(12, 7), dpi=80, facecolor='w', edgecolor='k')

    plt.plot(x, y, label=r'$\lambda =-1$')

    y = crra_utility(x, -0.2)
    plt.plot(x, y, label=r'$\lambda =-0.2$')

    y = crra_utility(x, -0.001)
    plt.plot(x, y, label=r'$\lambda =0$')

    # y = crra_utility(x, 0.5)
    # plt.plot(x, y,  label=r'$\lambda =0.5$')

    y = crra_utility(x, 1)
    plt.plot(x, y, label=r'$\lambda =1$')
    plt.legend()
    plt.title('CRRA Utility Function ' r' U(x)=$\frac{ 1- x^{-\lambda}}{\lambda} $')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$U(x)$')
    # plt.xticks([])
    # plt.yticks([])
    plt.show()


def expected_utility(x):
    q = [1.2, 1, 0.9]
    p = [0.4, 0.3, 0.3]
    alpha = 2  # alpha =1 stands for half kelly

    z = 0
    for index in range(0, len(q)):
        y = x * q[index] + 1 - x
        z = z + p[index] * crra_utility(y, alpha)
    return -z


def main():
    x0 = np.array([0.2])

    res = minimize(expected_utility, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

    print(res.x)

    x = np.linspace(0.0, 3.0, num=300)

    y = expected_utility(x)

    plt.plot(x, -y)
    plt.legend()
    plt.title('CRRA Utility Function ' r' U(x)=$\frac{ 1- x^{-\lambda}}{\lambda} $')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$U(x)$')
    # plt.xticks([])
    # plt.yticks([])
    plt.show()


main()
