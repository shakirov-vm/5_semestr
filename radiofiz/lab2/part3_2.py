import numpy as np # импорт бибилиотеки numpy
import matplotlib.pyplot as plt # импорт модуля matplotlib.pyplot
from scipy import signal
from scipy.linalg import dft
from scipy import integrate
from scipy.fft import fft
import math

m_0 = 3
m_1 = -0.25

N = 32

rg = 100

k = np.arange(rg)
m = m_0
x_k = np.cos(2 * np.pi * m * k / N) + np.sin(2 * np.pi * m * k / N)

plt.figure(figsize = [8, 4])

plt.stem(k, x_k)
plt.title('дискретный сигнал $x[k]$')
plt.xlabel('k') 
plt.ylabel('x[k]') 

k_range = np.linspace(0, rg, 1000)
x_k_range = np.cos(2 * np.pi * m * k_range / N) + np.sin(2 * np.pi * m * k_range / N)
plt.plot(k_range, x_k_range)

plt.grid()
plt.show()

# вычислим значения ДПФ на одном периоде
X = np.array([1 / N * np.dot(x_k[:N], np.exp(-1j * 2 * np.pi * k[:N] * n / N)) for n in range(N)])

plt.stem(np.arange(N), X.real)
plt.title('Действительная часть ДПФ')
plt.ylabel('Re X(n)')
plt.xlabel('n')
plt.grid()
plt.show()

plt.stem(np.arange(N), X.imag)
plt.title('Мнимая часть ДПФ')
plt.ylabel('Im X(n)')
plt.xlabel('n')
plt.grid()
plt.show()

plt.figure(figsize = [8, 4])

plt.stem(np.arange(N), 1 / N * fft(x_k[:N]).real)
plt.title('Действительная часть ДПФ')
plt.ylabel('Re X(n)')
plt.xlabel('n')
plt.grid()
plt.show()

plt.stem(np.arange(N), 1 / N * fft(x_k[:N]).imag)
plt.title('Мнимая часть ДПФ')
plt.ylabel('Im X(n)')
plt.xlabel('n')
plt.grid()
plt.show()

# 

k = np.arange(rg)
m = m_0 + m_1
x_k = np.cos(2 * np.pi * m * k / N) + np.sin(2 * np.pi * m * k / N)

plt.figure(figsize = [8, 4])

plt.stem(k, x_k)
plt.title('дискретный сигнал x[k]')
plt.xlabel('k') 
plt.ylabel('x[k]')

k_range = np.linspace(0, rg, 1000)
x_k_range = np.cos(2 * np.pi * m * k_range / N) + np.sin(2 * np.pi * m * k_range / N)

plt.plot(k_range, x_k_range)

plt.grid()
plt.show()

plt.stem(np.arange(N), 1 / N * fft(x_k[:N]).real)
plt.title('Действительная часть ДПФ')
plt.ylabel('Re X(n)')
plt.xlabel('n')
plt.grid()
plt.show()

plt.stem(np.arange(N), 1 / N * fft(x_k[:N]).imag)
plt.title('Мнимая часть ДПФ')
plt.ylabel('Im X(n)')
plt.xlabel('n')
plt.grid()
plt.show()

     
