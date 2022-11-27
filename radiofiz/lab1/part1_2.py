import numpy as np                # импорт бибилиотеки numpy
import matplotlib.pyplot as plt   # импорт модуля matplotlib.pyplot

# wav file

import scipy.io.wavfile # импорт модуля scipy.io.wavfile
import IPython.display as disp
import sys
#from IPython.display import Audio
disp.Audio('./dataset_lab1/glockenspiel.wav')

fs, x = scipy.io.wavfile.read('./dataset_lab1/glockenspiel.wav') # чтение аудиофайла 

print('Частота дискретизации: ', fs)
print('Битная глубина записи: ', x.dtype)
print('Число уровней квантования: ', 2 ** 32)
print('По графику число отсчётов (если брать 8000:8020):\n0.000125 с / 12 =', 0.000125 / 12, 'с')
print('А при нашей частоте дискретизации:', 1 / fs, 'с')
print('У нас размер массива данных:', x.size, ', тогда при нашей частоте дискретизации\nдлина нашей записи:', x.size / fs, 'с')
print('Размер нашего файла должен быть:', x.size * 4 / 1024, 'KB')

x1=x[0:100]                     # выбор наблюдаемого диапазона
k=np.arange(x1.size)               # отсчеты по времени
# Построение графиков 
plt.figure(figsize=[16, 4])         # создание полотна размером шириной 8 X 4 дюйма
plt.plot(k/fs, x1, 'b.')           # построение графика цифрового сигнала точками точками
plt.grid()                             
plt.xlabel("$t$, c")                      
plt.ylabel("$x[k]$")             
plt.tight_layout()

plt.show()






