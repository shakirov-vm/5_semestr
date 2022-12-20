import numpy as np # импорт бибилиотеки numpy
import matplotlib.pyplot as plt # импорт модуля matplotlib.pyplot
from scipy import signal
from scipy.linalg import dft
from scipy import integrate

m_0 = 3
m_1 = -0.25

N = 32

def dtft(x, M = 2048):
  return np.arange(M) / M - 0.5, np.fft.fftshift(np.fft.fft(x, M))

k = np.arange(N)
n = np.arange(N)
x = np.sin((2 * np.pi * m_0 * k) / N) + np.sin((2 * np.pi * (m_0 + 0.25) * k) / N)

plt.figure(figsize = [8, 4])

nu, X = dtft(x)

plt.plot(nu, abs(X))
plt.stem(np.arange(N) / N - 0.5, abs(np.fft.fftshift(np.fft.fft(x, N))))

plt.xlim([-0.5, 0.5])

plt.title('ДВПФ и ДПФ x[k]')
plt.ylabel('|X(nu)|, |X[n]|')

plt.grid()

plt.figure(figsize = [8, 4])

nu, X = dtft(x)
plt.plot(nu, abs(X))

N_z = 48
N_end = N + N_z

plt.stem(np.arange(N_end) / N_end - 0.5, abs(np.fft.fftshift(np.fft.fft(x, N_end))))

plt.xlim([-0.5, 0.5])

plt.title('ДВПФ и ДПФ x[k] с нулевыми отсчётами')
plt.ylabel('|X(nu)|, |X[n]|')

plt.grid()
plt.show()