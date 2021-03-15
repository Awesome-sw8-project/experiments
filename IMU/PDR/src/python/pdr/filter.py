from __future__ import division
import numpy as np

# General class for low-pass and high-pass filtering.
class Filter:
    # Low-pass filter non-convoluted (whatever that means).
    # sampleRate is the number of samples per second measured in kHz.
    # sampleNumber is simply the number of samples.
    @staticmethod
    def __lowpassfilter_nonconvoluted(sample_rate, sample_number):
        fc = 1 / sample_rate
        b = 4 / sample_number    # Approx. 4 / N.
        N = int(np.ceil(4 / b))
        if not N % 2: N += 1
        n = np.arange(N)
        h = np.sinc(2 * fc * (n - (N - 1) / 2))
        w = np.blackman(N)
        h = h * w
        return (h / np.sum(h), N)

    # High-pass filter.
    @staticmethod
    def highPassFilter(signal, sample_rate, sample_number = 50):
        non_convoluted = Filter.__lowpassfilter_nonconvoluted(sample_rate, sample_number)
        h = non_convoluted[0]
        N = non_convoluted[1]
        h = -h
        h[(N - 1) // 2] += 1
        return np.convolve(signal, h)

    # Low-pass filter.
    @staticmethod
    def lowPassFilter(signal, sample_rate, sample_number = 50):
        non_convoluted = Filter.__lowpassfilter_nonconvoluted(sample_rate, sample_number)[0]
        return np.convolve(signal, non_convoluted)
