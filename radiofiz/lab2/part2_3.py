import numpy as np # импорт бибилиотеки numpy
import matplotlib.pyplot as plt # импорт модуля matplotlib.pyplot
from scipy import signal
from scipy.linalg import dft
from scipy import integrate

x_k = np.array([5, 3, 2, 0, 6, -7, 4, -6])

m = -1

num = np.arange(8)

plt.figure(figsize = [8, 4])
plt.title("First")
plt.stem(num, x_k)

plt.grid()

X  = np.fft.fft(x_k)
Y = np.array([np.exp(-1j * 2 * np.pi / 8 * m * n) * X[n] for n in range(8)])
Y_fft = np.fft.ifft(Y)

plt.figure(figsize = [8, 4])
plt.title("Second")
plt.stem(num, Y_fft)

plt.grid()
plt.show()