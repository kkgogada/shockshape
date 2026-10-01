"""Python mirror of src/geometry.js (sharp-edged CST, handle-height design vector).

The browser app and the offline pipeline must agree exactly on what a design
vector means, or the surrogate learns the wrong airfoils. tests/test_pipeline.py
checks this file against fixtures exported from the JS implementation.
"""
from math import comb

import numpy as np

HANDLE_X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
N_HANDLES = len(HANDLE_X)
N_SHAPE = 2 * N_HANDLES
ORDER = N_HANDLES - 1


def _bernstein(i, n, x):
    return comb(n, i) * x**i * (1 - x) ** (n - i)


def _class(x):
    return x * (1 - x)


_M = np.array([[_class(xh) * _bernstein(i, ORDER, xh) for i in range(N_HANDLES)] for xh in HANDLE_X])
_M_INV = np.linalg.inv(_M)


def basis(x):
    """Cardinal basis L_j(x) (shape (len(x), 5)) so that y(x) = L @ heights."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    B = np.stack([_bernstein(i, ORDER, x) for i in range(N_HANDLES)], axis=1)
    return (_class(x)[:, None] * B) @ _M_INV


def surfaces(shape, x):
    shape = np.asarray(shape, dtype=float)
    L = basis(x)
    return L @ shape[:N_HANDLES], L @ shape[N_HANDLES:]


def cosine_stations(n):
    k = np.arange(n)
    return 0.5 * (1 - np.cos(np.pi * k / (n - 1)))


def coordinates(shape, n=201):
    """Closed airfoil loop TE -> upper -> LE -> lower -> TE (Selig order), unit chord."""
    x = cosine_stations(n)
    yu, yl = surfaces(shape, x)
    xs = np.concatenate([x[::-1], x[1:]])
    ys = np.concatenate([yu[::-1], yl[1:]])
    return np.stack([xs, ys], axis=1)


def max_thickness(shape):
    x = cosine_stations(401)
    yu, yl = surfaces(shape, x)
    return float(np.max(yu - yl))
