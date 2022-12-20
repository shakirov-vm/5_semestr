import numpy as np # импорт бибилиотеки numpy
import matplotlib.pyplot as plt # импорт модуля matplotlib.pyplot
from scipy import signal
from scipy.linalg import dft
from scipy import integrate

N = 6
L = 4
nu_0 = 0.1

points_quantity = 1000

x_k = np.ones(N)

plt.figure(figsize = [8, 4])

nu = np.arange(-0.5, 0.5, 1 / points_quantity)
dvpf = np.fft.fftshift(np.fft.fft(x_k, points_quantity))

dvpf_abs = np.abs(dvpf)

plt.plot(nu, dvpf_abs)

plt.title('Модуль')
plt.xlabel('Частота')
plt.ylabel('Модуль')

plt.grid()
plt.show()

x_k_shift = [np.exp(1j * 2 * np.pi * nu_0 * k) * x_k[k] for k in range(N)]

plt.figure(figsize = [8, 4])

nu = np.arange(-0.5, 0.5, 1 / points_quantity)
dvpf = np.fft.fftshift(np.fft.fft(x_k_shift, points_quantity))

dvpf_abs = np.abs(dvpf)

print("X = e^(-j * pi * (nu - nu_0)(N - 1)) * sin(pi * (nu - nu_0) * n) / sin(pi * (nu - nu_0))")

plt.plot(nu, dvpf_abs)

plt.title('Модуль')
plt.xlabel('Частота')
plt.ylabel('Модуль')

plt.grid()
plt.show()
