import numpy as np # импорт бибилиотеки numpy
import matplotlib.pyplot as plt # импорт модуля matplotlib.pyplot
import scipy.integrate as integrate # импорт модуля численного интегрирования

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

def sin_with_boxcar(t, tau):                            
    if 0<=t<=tau:
        # случай прямоугольного окна
        return np.sin(2*np.pi*f0*t) 
    else:
        return 0.0

def sin_with_hann(t, tau):                            
    if 0<=t<=tau:
        # случай окна Ханна
        return np.sin(2*np.pi*f0*t) * (0.5+0.5*np.cos(np.pi*(t-tau/2)/(tau/2)))
    else:
        return 0.0

f0=60*1e3       # 60 кГц
tau=100*1e-6    # 100 мкс

f_band=np.linspace(-2*f0, 2*f0, 2000) 
freq = f_band/1e3

t_band=np.linspace(-0.5*tau, 1.5*tau, 1024)

plt.figure(figsize=[8, 4])
plt.plot(t_band*1e6, [sin_with_boxcar(t, tau) for t in t_band])
plt.title("Сигнал с прямоугольным окном")
plt.xlabel("Время t, мкс")
plt.ylabel("$x(t)$, В")
plt.tight_layout() 
plt.grid()

spectr_boxcar = fourier_transform(signal=sin_with_boxcar, f_band=f_band, tau=tau, t1=-2*tau, t2=2*tau, res_type="abs")*1e6
main_freq_inx = np.argmax(spectr_boxcar)
print('В случае прямоугольного окна частота %0.3f кГц' %abs(freq[main_freq_inx]))

rigth_main_bound = main_freq_inx + 1
while spectr_boxcar[rigth_main_bound + 1] < spectr_boxcar[rigth_main_bound]:
    rigth_main_bound += 1

print('Ширина главного лепестка: %0.3f кГц' %(2 * abs(freq[rigth_main_bound] - freq[main_freq_inx])))

plt.figure(figsize=[8, 4])
plt.plot(f_band/1e3, spectr_boxcar)
plt.title("Спектр с прямоугольным окном")
plt.xlabel("Частота f, кГц")
plt.ylabel("$|X(f)|$,  мкВ / Гц")
plt.tight_layout() 
plt.grid()

plt.figure(figsize=[8, 4])
plt.plot(t_band*1e6, [sin_with_hann(t, tau) for t in t_band])
plt.title("Сигнал с окном Ханна")
plt.xlabel("Время t, мкс")
plt.ylabel("$x(t)$, В")
plt.tight_layout()
plt.grid()


spectr_hann = fourier_transform(signal=sin_with_hann, f_band=f_band, tau=tau, t1=-2*tau, t2=2*tau, res_type="abs")*1e6
main_freq_inx = np.argmax(spectr_hann)
print('В случае окна Ханна частота %0.3f кГц' %abs(freq[main_freq_inx]))

rigth_main_bound = main_freq_inx + 1
while spectr_hann[rigth_main_bound + 1] < spectr_hann[rigth_main_bound]:
    rigth_main_bound += 1

print('Ширина главного лепестка: %0.3f кГц' %(2 * abs(freq[rigth_main_bound] - freq[main_freq_inx])))

plt.figure(figsize=[8, 4])
plt.plot(f_band/1e3, spectr_hann)
plt.title("Спектр с окном Ханна")
plt.xlabel("Частота f, кГц")
plt.ylabel("$|X(f)|$,  мкВ / Гц")
plt.tight_layout() 
plt.grid()

plt.show()