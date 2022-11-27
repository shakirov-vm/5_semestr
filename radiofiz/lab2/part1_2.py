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

place = 1

end_seq = 4 * x_k.size - 4
while place < end_seq:
	for i in range(3):
 		x_k = np.insert(x_k, place, 0)
	place += 4

plt.figure(figsize = [8, 4])

nu = np.arange(-0.5, 0.5, 1 / points_quantity)
dvpf = np.fft.fftshift(np.fft.fft(x_k, points_quantity))

dvpf_abs = np.abs(dvpf)

plt.plot(nu, dvpf_abs)

plt.xlim ([-0.5, 0.5])

plt.title('Модуль')
plt.xlabel('Частота')
plt.ylabel('Модуль')

plt.grid()
plt.show()