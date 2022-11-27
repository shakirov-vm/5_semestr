import numpy as np # импорт бибилиотеки numpy
import matplotlib.pyplot as plt # импорт модуля matplotlib.pyplot
from scipy import signal
from scipy.linalg import dft
from scipy import integrate

N = 6
L = 4
nu_0 = 0.1

x_k = np.array([5, 3, 2, 0, 6, -7, 4, -6])

m = -1

# 8-dim basis
matr_dft = dft(8) @ x_k  # Compute the DFT of x_k
fft_dft  = np.fft.fft(x_k)

print('difference:', max(abs(matr_dft - fft_dft)))
