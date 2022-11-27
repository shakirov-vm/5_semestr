

import numpy as np              
import matplotlib.pyplot as plt   

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore")

def quantize_uniform(x, quant_min=-1, quant_max=1, quant_level=5):
    """Uniform quantization approach
    From: Müller M. Fundamentals of music processing: Audio, analysis, algorithms, applications. – Springer, 2015.
    Notebook: C2S2_DigitalSignalQuantization.ipynb
    Args:
        x: Original signal
        quant_min: Minimum quantization level
        quant_max: Maximum quantization level
        quant_level: Number of quantization levels
    Returns:
        x_quant: Quantized signal
    """
    x_normalize = (x-quant_min) * (quant_level-1) / (quant_max-quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)
    x_quant = (x_normalize_quant) * (quant_max-quant_min) / (quant_level-1) + quant_min
    return x_quant


N = 30                                    # Число отсчетов по времени
f0 = 400.0                                # Частота синусоиды в Гц
fs = 2000.0                               # Частота дискретизации в Гц
k = np.arange(N)                          # Mассив времен k от 0 до N-1 с шагом 1
x = np.sin(2*np.pi*(f0/fs)*k)             # Последовательность x[k]
num_levels = 50                           # число уровней квантования

# Построение графиков аналогового и дискретизованного сигнала
plt.figure(figsize = [12, 4])           
t = np.linspace(0, N/fs, num = 1024)        # создание массива времен t (1024 значения от 0 до N*fs)
plt.plot(t, np.sin(2*np.pi*f0*t), 'g', label = 'аналоговый сигнал $x(t)$') # построение графика x(t) (точки соединяются линиями)
plt.stem(k/fs, x, 'b', 'bo', label = 'дискретизованный сигнал $x[k]$') # построение графика функции дискретного времени x[k]

plt.grid()                              
plt.xlabel("$t$, c")                    
plt.ylabel("$x(t), x[k]$")              
plt.title("Аналоговый и дискретизованный сигналы")
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.tight_layout()                      # автоматическая корректировка расположения осей графика

plt.show()

# Построение графика цифрового сигнала
plt.figure(figsize = [12, 4])             
t = np.linspace(0, N/fs, num = 1024)        # создание массива времен t (1024 значения от 0 до N*fs)
h = quantize_uniform(x, quant_min=-1, quant_max=1, quant_level=5)
plt.plot(t, np.sin(2*np.pi*f0*t), 'g', label = 'аналоговый сигнал $x(t)$') # построение графика x(t) (точки соединяются линиями)
plt.stem(k/fs, h, 'b', 'bo', label = 'цифровой сигнал $x[k]$') # построение графика функции дискретного времени x[k]

plt.grid()                            
plt.xlabel("$t$, c")                   
plt.ylabel("$x(t), x[k]$")             
plt.title("Аналоговый и цифровой сигналы")
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.tight_layout()                      # автоматическая корректировка расположения осей графика

plt.show()

print('По второму графику видно, что проквантованный дискретный сигнал не попадает по значениям в аналоговый')

# ошибка квантования в зависимости от кол-ва уровней квантования

levels = np.array(range(5, 200, 1))

res = np.zeros(levels.size)

for i in range(levels.size):

    quanted = quantize_uniform(x, quant_min = -1, quant_max = 1, quant_level = i)
    res[i] = abs(x[1] - quanted[1])

plt.figure(figsize=[12, 4]) 
plt.plot(levels, res)
plt.xlabel("Кол-во уровней квантования")
plt.ylabel("Ошибка")
plt.title('$eps[k]=|x[k]-y[k]|$')
plt.ylim(0, 0.08)
plt.grid()

plt.show()

print('Очевидно, с ростом кол-ва уровней квантования, ошибка уменьшается')