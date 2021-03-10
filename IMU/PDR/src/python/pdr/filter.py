from __future__ import division
import numpy as np

# General class for low-pass and high-pass filtering.
class Filter:
    # Low-pass filter non-convoluted (whatever that means).
    @staticmethod
    def __lowPassFilterNonConvoluted(sampleRate):
        fc = sampleRate / 10
        b = 0.08
        N = int(np.ceil(4 / b))
        if not N % 2: N += 1
        n = np.arange(N)
        h = np.sinc(2 * fc * (n - (N - 1) / 2))
        w = np.blackman(N)
        h = h * w
        return (h / np.sum(h), N)

    # High-pass filter.
    @staticmethod
    def highPassFilter(signal, sampleRate):
        nonConvoluted = Filter.__lowPassFilterNonConvoluted(sampleRate)
        h = nonConvoluted[0]
        N = nonConvoluted[1]
        h = -h
        h[(N - 1) // 2] += 1
        return np.convolve(signal, h)

    # Low-pass filter.
    @staticmethod
    def lowPassFilter(signal, sampleRate):
        nonConvoluted = Filter.__lowPassFilterNonConvoluted(sampleRate)[0]
        return np.convolve(signal, nonConvoluted)
