from numpy import linspace, sqrt, sin, exp, convolve, abs, where

def rectangular(x):
    return where(abs(x) <= 0.5, 1, 0)
