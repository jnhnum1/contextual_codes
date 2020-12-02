from math import log
from scipy.optimize import minimize, minimize_scalar


def arg_min_scalar(objective, **kwargs):
    """
    Takes the same keyword arguments as minimize_scalar, but returns
    just the input that achieves the minimum value.
    """
    return minimize_scalar(objective, **kwargs).x


def arg_max_scalar(objective, **kwargs):
    """
    Analogous to arg_min_scalar.
    """
    return arg_min_scalar(lambda x: -objective(x), **kwargs)


def min_scalar(objective, **kwargs):
    """
    Takes the same keyword arguments as minimize_scalar, but returns
    just the minimum value achieved.
    """
    result = minimize_scalar(objective, **kwargs)
    return result.fun


def max_scalar(objective, **kwargs):
    """
    Analogous to min_scalar.
    """
    return -min_scalar(lambda x: -objective(x), **kwargs)


# Entropy and the like
def lg(x):
    """
    Returns log_2(x).
    """
    return log(x, 2)


def H(p):
    """
    Binary entropy function.
    """
    if p > 1 or p < 0:
        raise ValueError(f"Invalid invocation: H({p})")
    if p == 0 or p == 1:
        return 0
    return -p * lg(p) - (1 - p) * lg(1 - p)


def average(x, y):
    return x + (y - x) / 2   # prevent weird precision issues or overflows


def inverse(f, a, b, num_iters=64):
    """
    For a function f that is monotonically increasing on the interval (a, b),
    returns the function f^{-1}
    """
    if a >= b:
        raise ValueError(f"Invalid interval ({a}, {b})")

    def g(y):
        if y > f(b) or y < f(a):
            raise ValueError(f"Invalid image ({y})")
        lower = a
        upper = b
        for _ in range(num_iters):
            mid = average(lower, upper)
            if f(mid) < y:
                lower = mid
            elif f(mid) > y:
                upper = mid
            else:
                return mid
        return mid

    return g


Hinv = inverse(H, 0.0, 0.5)