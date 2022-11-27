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
x_k_convolution = signal.convolve(x_k, x_k)

plt.figure(figsize = [8, 4])

nu = np.arange(-0.5, 0.5, 1 / points_quantity)
dvpf = np.fft.fftshift(np.fft.fft(x_k_convolution, points_quantity))

dvpf_abs = np.abs(dvpf)

plt.plot(nu, dvpf_abs)

plt.title('Модуль')
plt.xlabel('Частота')
plt.ylabel('Модуль')

plt.grid()
plt.show()

print('X(0):', np.abs(dvpf[500]))

left = 500
while dvpf_abs[left] > dvpf_abs[left - 1]:
    left -= 1

print('Ширина главного лепестка на нулевом уровне:', np.abs(nu[left]) * 2)
print('Энергия:', integrate.simps(dvpf_abs ** 2, x = nu))