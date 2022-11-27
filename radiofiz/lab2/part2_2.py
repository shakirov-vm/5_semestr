import numpy as np # импорт бибилиотеки numpy
import matplotlib.pyplot as plt # импорт модуля matplotlib.pyplot
from scipy import signal
from scipy.linalg import dft
from scipy import integrate

x_k = np.array([5, 3, 2, 0, 6, -7, 4, -6])

m = -1

num = np.arange(8)
fft_dft  = np.fft.fft(x_k)

plt.figure(figsize = [8, 4])

plt.subplot(3, 3, 1)
plt.stem(num, fft_dft.real)
plt.ylabel('Re(X[n])')
plt.grid()

plt.subplot(3, 3, 3)
plt.stem(num, fft_dft.imag)
plt.ylabel('Im(X[n])')
plt.grid()

plt.subplot(3, 3, 7)
plt.stem(num, abs(fft_dft))
plt.ylabel('|X[n]|')
plt.grid()

plt.subplot(3, 3, 9)
plt.stem(num, np.angle(fft_dft))
plt.ylabel('phase(X[n])')
plt.grid()
plt.show()