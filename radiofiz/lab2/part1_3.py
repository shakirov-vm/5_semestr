import numpy as np # импорт бибилиотеки numpy
import matplotlib.pyplot as plt # импорт модуля matplotlib.pyplot
from scipy import signal
from scipy.linalg import dft
from scipy import integrate

N = 6
L = 4
nu_0 = 0.1

points_quantity = 1000

x_k_old = np.ones(N)
x_k = np.array([i * x_k_old[i] for i in range (N)])

nu = np.arange(-0.5, 0.5, 1 / points_quantity)
dvpf = np.fft.fftshift(np.fft.fft(x_k, points_quantity))

dvpf_abs = np.abs(dvpf)

plt.figure(figsize = [8, 4])
plt.plot(nu, dvpf_abs)

plt.xlim ([-0.5, 0.5])

plt.title('Модуль')
plt.xlabel('Частота')
plt.ylabel('Модуль')

plt.grid()
plt.show()

x_k_diff = 1j / (2 * np.pi) * (1j * 2 * np.pi * N * np.exp(-1j * 2 * np.pi * nu * N) * \
				 (1 - np.exp(-1j * 2 * np.pi * nu)) - (1 - np.exp(-1j * 2 * np.pi * nu * N)) * \
				 1j * 2 * np.pi * np.exp(-1j * 2 * np.pi * nu)) / ((1 - np.exp(-1j * 2 * np.pi * nu)) ** 2)
x_k_diff[500] /= 2

plt.figure (figsize = [8, 4])
plt.plot(nu, abs(x_k_diff))

plt.xlim ([-0.5, 0.5])

plt.title('Модуль')
plt.xlabel('Частота')
plt.ylabel('Модуль')

plt.grid()
plt.show()