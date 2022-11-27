import numpy as np # импорт бибилиотеки numpy
import matplotlib.pyplot as plt # импорт модуля matplotlib.pyplot
import scipy.integrate as integrate # импорт модуля численного интегрирования
from math import cos, pi, log

def integrate_function(t, func, f, tau, real_part=True):
    # Подынтегральное выражение для использованиия в функции integrate.quad
    # t - время
    # func - функция, задающая импульс
    # f - частота
    # tau - константа, используемая для задания длительности импульса
    if real_part:
        return func(t, tau)*np.cos(-2*np.pi*f*t)  # действительная часть интеграла
    else:
        return func(t, tau)*np.sin(-2*np.pi*f*t)  # мнимая часть интеграла

def fourier_transform(signal, f_band, tau, t1, t2, res_type='abs'):
    # вычисление преобразования Фурье для набора частот
    # signal - функция от t и tau, задающая сигнал во временной области 
    # f_band - набор частот, для которых вычисляется преобразование Фурье
    # tau - константа, используемая для задания длительности импульса
    # t1 момент начала сигнала
    # t2 момент завершения сигнала
    # тип возвращаемого значения:
    # res_type='abs' - |X(f)|
    # res_type='Re' - Re X(f)
    # res_type='Im' - Im X(f)
    if res_type=="abs":
        Re=np.array([integrate.quad(integrate_function, t1, t2, args=(signal, f, tau, True))[0] for f in f_band])
        Im=np.array([integrate.quad(integrate_function, t1, t2, args=(signal, f, tau, False))[0] for f in f_band])
        return abs(Re+1j*Im)
    elif res_type=="Re":
        Re=np.array([integrate.quad(integrate_function, t1, t2, args=(signal, f, tau, True))[0] for f in f_band])
        return Re
    elif res_type=="Im":
        Im=np.array([integrate.quad(integrate_function, t1, t2, args=(signal, f, tau, False))[0] for f in f_band])
        return Im

tau = 500e-6 # 500 мкс
f_band=np.linspace(-8/tau, 8/tau, 500) # 500 - число точек в диапазоне, в которых вычисляется X(f)

def boxcar(t, tau):
    if abs(t)<tau/2:
        return 1.0                 
    else:
        return 0.0  

def triangle(t, tau):
    if abs(t)<tau/2:
        return 1.0 - abs(t)*2 / tau               
    else:
        return 0.0  

def hann(t, tau):
    if abs(t)<tau/2:
        return 0.5 * (1 + cos(2*pi * t/tau))             
    else:
        return 0.0    


def print_sig_and_prec(function):

    t_band=np.linspace(-2*tau, 2*tau, 1024)
    plt.figure(figsize=[6, 4])
    plt.plot(t_band*1e6, [function(t, tau) for t in t_band])
    plt.xlabel("Время t, мкс")
    plt.ylabel("$x(t)$, В")
    plt.title("Сигнал")
    plt.tight_layout() 
    plt.grid()

    plt.show()

    plt.figure(figsize=[6, 4])
    plt.plot(f_band/1e3, fourier_transform(signal=function, f_band=f_band, tau=tau, t1=-tau/2, t2=tau/2, res_type="abs")*1e6)
    plt.xlabel("Частота f, кГц")
    plt.ylabel("$|X(f)|$,  мкВ / Гц")
    plt.title("Спектр")
    plt.tight_layout() 
    plt.grid()

    spectr = fourier_transform(signal=function, f_band=f_band, tau=tau, t1=-tau/2, t2=tau/2, res_type="abs")*1e6
    freq = f_band/1e3

# это индексы
    main_petal = np.argmax(spectr)

    rigth_main_bound = main_petal + 1
    while spectr[rigth_main_bound + 1] < spectr[rigth_main_bound]:
        rigth_main_bound += 1
    first_rigth_petal = rigth_main_bound + 1
    while spectr[first_rigth_petal + 1] > spectr[first_rigth_petal]:
        first_rigth_petal += 1

    left_main_bound = main_petal - 1
    while spectr[left_main_bound - 1] < spectr[left_main_bound]:
        left_main_bound -= 1
    first_left_petal = left_main_bound - 1
    while spectr[first_left_petal - 1] > spectr[first_left_petal]:
        first_left_petal -= 1

    print('Главный:', spectr[main_petal], ', боковой:', spectr[first_rigth_petal])

    print('Уровень первого бокового лепестка относительно главного: %0.3f dB' %(20 * log(abs(spectr[first_rigth_petal] / spectr[main_petal]), 10)))
    print('Ширина главного лепестка: %0.3f кГц' %(freq[rigth_main_bound] - freq[left_main_bound]))

    plt.show()

print('Прямоугольный:')
print_sig_and_prec(boxcar)
print('Треугольный:')
print_sig_and_prec(triangle)
print('Окно Ханна:')
print_sig_and_prec(hann)
