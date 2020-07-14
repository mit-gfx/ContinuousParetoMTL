# Source code for ICML submission #640 "Efficient Continuous Pareto Exploration in Multi-Task Learning"
import numpy as np

from common import *

class Zdt2Variant(object):
    def __init__(self):
        self.n = 3
        self.m = 2

        self.eval_f_cnt = 0
        self.eval_grad_cnt = 0
        self.eval_hvp_cnt = 0

    def reset_count(self):
        self.eval_f_cnt = 0
        self.eval_grad_cnt = 0
        self.eval_hvp_cnt = 0

    def __remap(self, x):
        x = ndarray(x).ravel()
        assert x.size == self.n
        x2 = np.zeros(self.n)
        x2[0] = np.sin(x[0] + x[1] ** 2 + x[2] ** 2) * 0.5 + 0.5
        s = np.sum(x[1:] ** 2)
        x2[1:] = 0.5 * np.cos(s) + 0.5
        return x2

    def __remap_grad(self, x):
        x = ndarray(x).ravel()
        assert x.size == self.n
        jac = np.zeros((self.n, self.n))
        jac[0] = 0.5 * np.cos(x[0] + x[1] ** 2 + x[2] ** 2) * ndarray([1, 2 * x[1], 2 * x[2]])
        s = np.sum(x[1:] ** 2)
        g_s = np.zeros(self.n)
        g_s[1:] = 2 * x[1:]
        jac[1:] = -0.5 * np.sin(s) * g_s
        return jac

    def __remap_hess(self, x):
        x = ndarray(x).ravel()
        assert x.size == self.n
        hess = np.zeros((self.n, self.n, self.n))
        s = np.sum(x[1:] ** 2)
        hess[0, 0, 0] = 0.5 * -np.sin(x[0] + s)
        hess[0, 0, 1:] = -np.sin(x[0] + s) * x[1:]
        hess[0, 1:, 0] = hess[0, 0, 1:]
        for i in range(1, self.n):
            hess[0, i, i] = np.cos(x[0] + s) + -np.sin(x[0] + s) * 2 * x[i] ** 2
        for i in range(1, self.n):
            for j in range(i + 1, self.n):
                hess[0, i, j] = hess[0, j, i] = x[i] * -np.sin(x[0] + s) * 2 * x[j]
        g_s = np.zeros(self.n)
        g_s[1:] = 2 * x[1:]
        for i in range(1, self.n):
            hess[1:, :, i] = -0.5 * np.cos(s) * g_s[i] * g_s
            hess[1:, i, i] += -0.5 * np.sin(s) * 2
        return hess

    def f(self, x):
        self.eval_f_cnt += 1
        return self.__f(self.__remap(x))

    def __f(self, x):
        x = ndarray(x).ravel()
        assert x.size == self.n
        f1 = x[0]
        g = 1 + 9 / (self.n - 1) * np.sum(x[1:])
        f2 = g * (1 - (x[0] / g) ** 2)
        return ndarray([f1, f2])

    def grad(self, x):
        self.eval_grad_cnt += 1
        x_new = self.__remap(x)
        grad_x_new = self.__remap_grad(x)
        g1, g2 = self.__grad(x_new)
        return ndarray([g1.T @ grad_x_new, g2.T @ grad_x_new])

    def __grad(self, x):
        x = ndarray(x).ravel()
        assert x.size == self.n

        g1 = np.zeros(self.n)
        g1[0] = 1

        grad_g = np.zeros(self.n)
        grad_g[1:] = 9 / (self.n - 1)
        g = 1 + 9 / (self.n - 1) * np.sum(x[1:])

        g2 = grad_g * (1 - (x[0] / g) ** 2)
        g2[0] += -2 * x[0] / g
        g2[1:] += 2 * (x[0] / g) ** 2 * grad_g[1:]

        return ndarray([g1, g2])

    def hess(self, x):
        x_new = self.__remap(x)

        g1, g2 = self.__grad(x_new)
        h1, h2 = self.__hess(x_new)
        g_remap = self.__remap_grad(x)
        h_remap = self.__remap_hess(x)

        # f(u, v), u = g(x1, x2), v = g(x1, x2).
        # df/dx1 = df/du * du/dx1 + df/dv * dv/dx1 = g1.dot(g_remap[:, 0])
        # ddf/dx1dx2 = (h1 @ g_remap[:, 1]).dot(g_remap[:, 0]) + g1.dot(h_remap[:, 0, 1])
        h1_remap = g_remap.T @ (h1 @ g_remap)
        h2_remap = g_remap.T @ (h2 @ g_remap)
        for i in range(self.n):
            h1_remap[i] += g1.T @ h_remap[:, i, :]
            h2_remap[i] += g2.T @ h_remap[:, i, :]
        return ndarray([h1_remap, h2_remap])

    def __hess(self, x):
        x = ndarray(x).ravel()
        assert x.size == self.n
        h1 = np.zeros((self.n, self.n))

        h2 = np.zeros((self.n, self.n))
        g = 1 + 9 / (self.n - 1) * np.sum(x[1:])
        grad_g = np.zeros(self.n)
        grad_g[1:] = 9 / (self.n - 1)

        # g2[0] = -2 * x[0] / g
        h2[0, 0] = -2 / g
        h2[0, 1:] = 18 * x[0] / g / g / (self.n - 1)
        # g2[1] = 9 / (n - 1) * (1 + (x[0] / g) ** 2)
        h2[1:, 0] = 18 * x[0] / g / g / (self.n - 1)
        h2[1:, 1:] = -2 / g * (9 / (self.n - 1) * x[0] / g) ** 2
        return ndarray([h1, h2])

    def hvp(self, x, alpha, v):
        self.eval_hvp_cnt += 1
        h1, h2 = self.hess(x)
        alpha = ndarray(alpha).ravel()
        assert alpha.size == self.m
        v = ndarray(v).ravel()
        assert v.size == self.n
        return ndarray(alpha[0] * h1 @ v + alpha[1] * h2 @ v)

    def sample_pareto_set(self):
        x = np.zeros(self.n)
        x[0] = np.random.uniform(-np.pi / 2, np.pi / 2) - np.pi
        theta = np.random.uniform(-np.pi, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        x[1] = np.sqrt(np.pi) * c
        x[2] = np.sqrt(np.pi) * s
        return ndarray(x)

    def plot_pareto_set(self, ax):
        x1_low, x1_high = -np.pi / 2 - np.pi, np.pi / 2 - np.pi
        r = np.sqrt(np.pi)
        theta = np.linspace(-np.pi, np.pi, 33)
        X2, X3 = r * np.cos(theta), r * np.sin(theta)
        X1 = np.outer(np.linspace(x1_low, x1_high, 9), np.ones(theta.size))

        face_color = np.zeros((X1.shape[0], X1.shape[1], 3))
        face_color[:] = [0.85, 0.93, 0.92]
        ax.plot_surface(X2, X3, X1, alpha=0.25, facecolors=face_color)

        ax.set_xlim([-2 * r, 2 * r])
        ax.set_ylim([-2 * r, 2 * r])
        ax.set_zlim([x1_low, x1_high])

        ax.set_xlabel('$x_2$')
        ax.set_ylabel('$x_3$')
        ax.set_zlabel('$x_1$')

    def plot_pareto_front(self, ax, label='Pareto front'):
        # Analytic Pareto front.
        f1 = np.linspace(0.0, 1.0, 101)
        f2 = 1 - f1 ** 2
        if label is None:
            ax.plot(f1, f2, 'k-.')
        else:
            ax.plot(f1, f2, 'k-.', label=label)
        ax.set_xlabel('$f_1$')
        ax.set_ylabel('$f_2$')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_aspect('equal')
        ax.grid(True)

if __name__ == '__main__':
    # Check gradients.
    problem = Zdt2Variant()
    n, m = 3, 2
    x0 = np.random.normal(size=n)
    for i in range(m):
        f = lambda x: problem.f(x)[i]
        g = lambda x: problem.grad(x)[i]
        check_grad(f, g, x0)
        h = lambda x : problem.hess(x)[i]
        check_hess(f, g, h, x0)

    # Check Pareto front.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig, ax = plt.subplots(1, 1)
    problem.plot_pareto_front(ax)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    problem.plot_pareto_set(ax)

    plt.show()
    plt.close()