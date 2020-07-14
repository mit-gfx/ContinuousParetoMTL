import sys
from itertools import combinations

import numpy as np
import torch


def _min_norm_element_from2(v1v1, v1v2, v2v2):
    """
    Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
    d is the distance (objective) optimzed
    v1v1 = <x1,x1>
    v1v2 = <x1,x2>
    v2v2 = <x2,x2>
    """
    if v1v2 >= v1v1:
        # Case: Fig 1, third column
        gamma = 0.999
        cost = v1v1
        return gamma, cost
    if v1v2 >= v2v2:
        # Case: Fig 1, first column
        gamma = 0.001
        cost = v2v2
        return gamma, cost
    # Case: Fig 1, second column
    gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2)
    # v2v2 - gamm * gamma * (v1 - v2)^2
    # cost = v2v2 - gamma * gamma * (v1v1 + v2v2 - 2 * v1v2)
    #      = v2v2 - gamma * (v2v2 - v1v2)
    cost = v2v2 + gamma * (v1v2 - v2v2)
    return gamma, cost


def _min_norm_2d(vecs):
    """
    Find the minimum norm solution as combination of two points
    This is correct only in 2D
    ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
    """
    dmin = None
    dps = vecs.matmul(vecs.t()).cpu().numpy()
    for i, j in combinations(range(len(vecs)), 2):
        c, d = _min_norm_element_from2(dps[i, i], dps[i, j], dps[j, j])
        if dmin is None:
            dmin = d
        if d <= dmin:
            dmin = d
            sol = [(i, j), c, d]
    return sol, dps


def _projection2simplex(y):
    """
    Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
    """
    m = len(y)
    sorted_y = np.flip(np.sort(y), axis=0)
    tmpsum = 0.0
    tmax_f = (np.sum(y) - 1.0) / m
    for i in range(m - 1):
        tmpsum += sorted_y[i]
        tmax = (tmpsum - 1) / (i + 1.0)
        if tmax > sorted_y[i + 1]:
            tmax_f = tmax
            break
    return np.maximum(y - tmax_f, np.zeros(y.shape))


def _next_point(cur_val, grad, n):
    proj_grad = grad - (np.sum(grad) / n)
    tm1 = -cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
    tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

    t = 1
    if len(tm1[tm1 > 1e-7]) > 0:
        t = np.min(tm1[tm1 > 1e-7])
    if len(tm2[tm2 > 1e-7]) > 0:
        t = min(t, np.min(tm2[tm2 > 1e-7]))

    next_point = proj_grad * t + cur_val
    next_point = _projection2simplex(next_point)
    return next_point


def find_min_norm_element(vecs, max_iter=250, stop_crit=1e-5):
    """
    Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
    as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
    It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j;
    the solution lies in (0, d_{i,j})
    Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
    """
    # Solution lying at the combination of two points
    init_sol, dps = _min_norm_2d(vecs.detach())

    n = len(vecs)
    sol_vec = np.zeros(n)
    sol_vec[init_sol[0][0]] = init_sol[1]
    sol_vec[init_sol[0][1]] = 1 - init_sol[1]

    if n < 3:
        # This is optimal for n=2, so return the solution
        return sol_vec, init_sol[2]

    iter_count = 0

    while iter_count < max_iter:
        grad_dir = -1.0 * np.dot(dps, sol_vec)
        new_point = _next_point(sol_vec, grad_dir, n)
        # Re-compute the inner products for line search
        v1v1 = 0.0
        v1v2 = 0.0
        v2v2 = 0.0
        for i in range(n):
            for j in range(n):
                v1v1 += sol_vec[i] * sol_vec[j] * dps[i, j]
                v1v2 += sol_vec[i] * new_point[j] * dps[i, j]
                v2v2 += new_point[i] * new_point[j] * dps[i, j]
        nc, nd = _min_norm_element_from2(v1v1, v1v2, v2v2)
        new_sol_vec = nc * sol_vec + (1 - nc) * new_point
        change = new_sol_vec - sol_vec
        if np.sum(np.abs(change)) < stop_crit:
            break
        sol_vec = new_sol_vec
    return sol_vec, nd


if __name__ == '__main__':
    import numpy as np
    import cvxpy as cp

    n = 10
    v1 = np.random.normal(size=n)
    v2 = np.random.normal(size=n)
    v1v1 = v1.dot(v1)
    v1v2 = v1.dot(v2)
    v2v2 = v2.dot(v2)
    # min \|c * x1 + (1 - c) * x2\|^2.
    # Ground truth.
    alpha = cp.Variable(2)
    V = np.array([v1, v2])  # V: 2 * n.
    objective = cp.Minimize(cp.sum_squares(V.T @ alpha))
    constraints = [alpha >= 0, cp.sum(alpha) == 1]
    prob = cp.Problem(objective, constraints)
    loss = prob.solve()

    gamma, cost = _min_norm_element_from2(v1v1, v1v2, v2v2)
    print('loss:', loss, 'alpha:', alpha.value)
    print('loss:', cost, 'alpha:', [gamma, 1 - gamma])
