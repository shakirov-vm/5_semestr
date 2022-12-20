import numpy as np
import matplotlib.pyplot as plt
import math as mth

def MNK(data_x, data_y):

	x_for_interpol  = data_x
	y_for_interpol  = data_y
	xy_for_interpol = x_for_interpol * y_for_interpol
	xx_for_interpol = x_for_interpol * x_for_interpol
	yy_for_interpol = y_for_interpol * y_for_interpol

	k = (np.mean(xy_for_interpol) - np.mean(x_for_interpol) * np.mean(y_for_interpol)) / (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2)
	k_err = np.sqrt(1 / 10 * np.fabs((np.mean(yy_for_interpol) - np.mean(y_for_interpol) ** 2) / \
	        (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2) - k ** 2))
	b = np.mean(y_for_interpol) - k * np.mean(x_for_interpol)
	b_err = k_err * np.sqrt(np.mean(xx_for_interpol))

	return k, k_err, b, b_err

x_arr = np.array([65, 83, 112, 134, 153])
y_arr = np.array([7.82, 10.93, 12.24, 15.37, 17.58])

k, k_err, b, b_err = MNK(x_arr, y_arr)

print(k, "+-", k_err)

plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linewidth = 1)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.title("Зависимость ЭДС индукции от напряжения на резисторе")
plt.ylabel("ЭДС индукции, мВ")
plt.xlabel("Напряжение на резисторе, мВ")
plt.errorbar(x_arr, y_arr, fmt = 'or')

points_x = np.linspace(x_arr[0], x_arr[x_arr.size - 1], 1000)
points_y = points_x * k + b

plt.plot(points_x, points_y, 'black', linewidth = 2)

plt.show()

x_arr = np.array([142, 142, 131])
y_arr = np.array([184, 180, 168])

k, k_err, b, b_err = MNK(x_arr, y_arr)

print(k, "+-", k_err)

plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linewidth = 1)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.title("Зависимость частоты от напряжения на резисторе")
plt.ylabel("Частота, МГц")
plt.xlabel("Напряжение на резисторе, мВ")
plt.errorbar(x_arr, y_arr, fmt = 'or')

points_x = np.linspace(x_arr[0], x_arr[x_arr.size - 1], 1000)
points_y = points_x * k + b

plt.plot(points_x, points_y, 'black', linewidth = 2)

plt.show()