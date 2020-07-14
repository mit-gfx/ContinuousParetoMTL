# Source code for ICML submission #640 "Efficient Continuous Pareto Exploration in Multi-Task Learning"
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')
    raise RuntimeError

def print_ok(*message):
    print('\033[92m', *message, '\033[0m')

def print_warning(*message):
    print('\033[93m', *message, '\033[0m')

def print_info(*message):
    print('\033[96m', *message, '\033[0m')

def ndarray(x):
    return np.asarray(x, dtype=np.float64)

# f: R^n -> R.
# grad: R^n -> R^n.
def check_grad(f, grad, x0, options={}):
    eps = 1e-6 if 'eps' not in options else options['eps']
    atol = 1e-6 if 'atol' not in options else options['atol']
    rtol = 1e-4 if 'rtol' not in options else options['rtol']

    analytic_g = grad(x0)
    n = x0.size
    for i in range(n):
        x0_pos = np.copy(x0)
        x0_pos[i] += eps
        f_pos = f(x0_pos)
        x0_neg = np.copy(x0)
        x0_neg[i] -= eps
        f_neg = f(x0_neg)
        numeric_g = (f_pos - f_neg) / 2 / eps
        assert np.isclose(numeric_g, analytic_g[i], atol=atol, rtol=rtol), \
            print_error('at x[{}]: {}, {}'.format(i, numeric_g, analytic_g[i]))

# f: R^n -> R.
# grad: R^n -> R^n.
# hess: R^n -> R^{n x n}.
def check_hess(f, grad, hess, x0, options={}):
    eps = 1e-6 if 'eps' not in options else options['eps']
    atol = 1e-6 if 'atol' not in options else options['atol']
    rtol = 1e-4 if 'rtol' not in options else options['rtol']

    analytic_h = hess(x0)
    n = x0.size
    for i in range(n):
        x0_pos = np.copy(x0)
        x0_pos[i] += eps
        x0_neg = np.copy(x0)
        x0_neg[i] -= eps
        g_pos = grad(x0_pos)
        g_neg = grad(x0_neg)
        numeric_h = (g_pos - g_neg) / 2 / eps
        assert np.allclose(numeric_h, analytic_h[i], atol=atol, rtol=rtol), \
            print_error('at x[{}]: {}, {}'.format(i, numeric_h, analytic_h[i]))

# True if x is dominated by y: y <= x and y != x.
def dominated(x, y, atol=1e-8):
    diff = x - y
    diff[np.isclose(diff, 0, atol=atol)] = 0
    return np.min(diff) >= 0 and np.max(diff) > 0

# Pareto stationary points -> pareto optimal points.
# xs: k x n matrix, i.e., k n-dimensional points.
# fs: k x m matrix, i.e., k m-dimensional f(points).
def filter_pareto_stationary_points(xs, fs, atol=1e-8):
    xs = np.asarray(xs)
    fs = np.asarray(fs)
    assert len(xs.shape) == 2 and len(fs.shape) == 2
    assert xs.shape[0] == fs.shape[0]

    x_filtered = []
    f_filtered = []
    for x, f in zip(xs, fs):
        if not np.any([dominated(f, f2, atol) for f2 in fs]):
            x_filtered.append(x)
            f_filtered.append(f)
    return np.asarray(x_filtered), np.asarray(f_filtered)

def compute_hypervolume(fs, ref_point):
    fs = ndarray(fs)
    if fs.size == 0: return 0
    assert len(fs.shape) == 2 and fs.shape[1] == 2, print_error('>2 dimensional cases are not implemented yet.')
    # Sort fs.
    idx = np.argsort(fs[:, 0])
    fs = fs[idx]
    hv = 0.0
    f_last = ref_point[1]
    for f1, f2 in fs:
        hv += (ref_point[0] - f1) * (f_last - f2)
        f_last = f2
    return hv

# Drawing functions.
# Fancy 3d arrow draing.
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def draw_arrow_3d(ax, head, tail, color, label=None):
    arrow = Arrow3D([tail[1], head[1]], [tail[2], head[2]], [tail[0], head[0]],
        mutation_scale=24, lw=8, arrowstyle='-|>', color=color, label=label)
    ax.add_artist(arrow)

def draw_arrow_2d(ax, head, tail, color, thickness, head_length, padding, label=None):
    arrow_length = np.linalg.norm(head - tail)
    if arrow_length < padding * 2 + head_length: return
    arrow_unit = (head - tail) / arrow_length
    tail_shifted = tail + arrow_unit * padding
    head_shifted = tail + arrow_unit * (arrow_length - padding - head_length)
    ax.arrow(tail_shifted[0], tail_shifted[1], head_shifted[0] - tail_shifted[0], head_shifted[1] - tail_shifted[1],
        width=thickness, head_length=head_length, fc=color, ec=color, label=label, alpha=0.5)