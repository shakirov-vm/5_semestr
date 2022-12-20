import numpy as np
import matplotlib.pyplot as plt
import math as mth
from scipy.interpolate import UnivariateSpline

grad = np.loadtxt('grad.txt', dtype = float)
grad_mono_x = grad[0]
grad_lamdbda_y = grad[1]
grad_mono_x_err = grad_mono_x * 0 + 5

grad_mono_x.sort()
grad_lamdbda_y.sort()

plt.figure()

spl = UnivariateSpline(grad_mono_x, grad_lamdbda_y)
spl_x = np.linspace(2150, 2610, 500)

freq_2200 = 3 * 1e8 / spl(2200) * 1e10
freq_2325 = 3 * 1e8 / spl(2325) * 1e10
freq_2550 = 3 * 1e8 / spl(2550) * 1e10

print("2200 degree is", spl(2200), "angstrem and", freq_2200 / 1000 / 1000 / 1000 / 1000, "THz")
print("2325 degree is", spl(2325), "angstrem and", freq_2325 / 1000 / 1000 / 1000 / 1000, "THz")
print("2550 degree is", spl(2550), "angstrem and", freq_2550 / 1000 / 1000 / 1000 / 1000, "THz")

plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linewidth = 2)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.title("Калибровка")
plt.xlabel("")
plt.ylabel("")
plt.errorbar(grad_mono_x, grad_lamdbda_y, xerr = grad_mono_x_err, fmt = 'o:k', ecolor = 'white')
plt.plot(spl_x, spl(spl_x), 'black', linewidth = 3)
#plt.scatter(grad_mono_x, grad_lamdbda_y, color = 'g', s = 20)
#plt.show()

# 2200

UI_2200 = np.loadtxt('2200.txt', dtype = float)
U_2200_x = UI_2200[0]
I_2200 = UI_2200[1]
I_sqrt_2200_y = I_2200
for i in range(I_2200.size): ## ???
	if I_2200[i] > 0:
		I_sqrt_2200_y[i] = I_2200[i]
	else:
		I_sqrt_2200_y[i] = -np.abs(I_2200[i])

U_2200_x_err = np.abs(U_2200_x) * 0.01
I_sqrt_2200_y_err = 1 / 2 * np.abs(I_sqrt_2200_y) * 0.01

plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linewidth = 2)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.title("2200")
plt.errorbar(U_2200_x, I_sqrt_2200_y, xerr = U_2200_x_err, yerr = I_sqrt_2200_y_err, fmt = 'o-r', ecolor = 'red')
#plt.show()

x_for_interpol  = U_2200_x[U_2200_x.size - 11:]
y_for_interpol  = I_sqrt_2200_y[U_2200_x.size - 11:]
xy_for_interpol = x_for_interpol * y_for_interpol
xx_for_interpol = x_for_interpol * x_for_interpol
yy_for_interpol = y_for_interpol * y_for_interpol

k_2200 = (np.mean(xy_for_interpol) - np.mean(x_for_interpol) * np.mean(y_for_interpol)) / (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2)
k_err_2200 = np.sqrt(1 / 10 * np.fabs((np.mean(yy_for_interpol) - np.mean(y_for_interpol) ** 2) / \
        (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2) - k_2200 ** 2))
b_2200 = np.mean(y_for_interpol) - k_2200 * np.mean(x_for_interpol)
b_err_2200 = k_err_2200 * np.sqrt(np.mean(xx_for_interpol))

print("For 2200:") 
print("\tk:", k_2200, "+-", k_err_2200)
print("\tb:", b_2200, "+-", b_err_2200)

V_0_2200 = b_2200 / k_2200
V_0_err_2200 = V_0_2200 * np.sqrt(k_err_2200 ** 2 / k_2200 ** 2 + b_err_2200 ** 2 / b_2200 ** 2)

print("Blocked for 2200:", V_0_2200, "+-", V_0_err_2200)

# 2550

UI_2550 = np.loadtxt('2550.txt', dtype = float)
U_2550_x = UI_2550[0]
I_2550 = UI_2550[1]
I_sqrt_2550_y = I_2550
for i in range(I_2550.size): ## ???
	if I_2550[i] > 0:
		I_sqrt_2550_y[i] = I_2550[i]
	else:
		I_sqrt_2550_y[i] = -np.abs(I_2550[i])

U_2550_x_err = np.abs(U_2550_x) * 0.01
I_sqrt_2550_y_err = 1 / 2 * np.abs(I_sqrt_2550_y) * 0.01

plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linewidth = 2)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.title("2550")
plt.errorbar(U_2550_x, I_sqrt_2550_y, xerr = U_2550_x_err, yerr = I_sqrt_2550_y_err, fmt = 'o-r', edgecolor=None)
#plt.show()

x_for_interpol  = U_2550_x[U_2550_x.size - 6:]
y_for_interpol  = I_sqrt_2550_y[U_2550_x.size - 6:]
xy_for_interpol = x_for_interpol * y_for_interpol
xx_for_interpol = x_for_interpol * x_for_interpol
yy_for_interpol = y_for_interpol * y_for_interpol

k_2550 = (np.mean(xy_for_interpol) - np.mean(x_for_interpol) * np.mean(y_for_interpol)) / (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2)
k_err_2550 = np.sqrt(1 / 10 * np.fabs((np.mean(yy_for_interpol) - np.mean(y_for_interpol) ** 2) / \
        (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2) - k_2550 ** 2))
b_2550 = np.mean(y_for_interpol) - k_2550 * np.mean(x_for_interpol)
b_err_2550 = k_err_2550 * np.sqrt(np.mean(xx_for_interpol))

print("For 2550:") 
print("\tk:", k_2550, "+-", k_err_2550)
print("\tb:", b_2550, "+-", b_err_2550)

V_0_2550 = b_2550 / k_2550
V_0_err_2550 = V_0_2550 * np.sqrt(k_err_2550 ** 2 / k_2550 ** 2 + b_err_2550 ** 2 / b_2550 ** 2)

print("Blocked for 2550:", V_0_2550, "+-", V_0_err_2550)

# 2325

UI_2325 = np.loadtxt('2325.txt', dtype = float)
U_2325_x = UI_2325[0]
I_2325 = UI_2325[1]
I_sqrt_2325_y = I_2325
for i in range(I_2325.size): ## ???
	if I_2325[i] > 0:
		I_sqrt_2325_y[i] = I_2325[i]
	else:
		I_sqrt_2325_y[i] = -np.abs(I_2325[i])

U_2325_x_err = np.abs(U_2325_x) * 0.01
I_sqrt_2325_y_err = 1 / 2 * np.abs(I_sqrt_2325_y) * 0.01

plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linewidth = 2)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.title("2325")
plt.errorbar(U_2325_x, I_sqrt_2325_y, xerr = U_2325_x_err, yerr = I_sqrt_2325_y_err, fmt = 'o-r', ecolor = 'red')
plt.show()

x_for_interpol  = U_2325_x[U_2325_x.size - 8:]
y_for_interpol  = I_sqrt_2325_y[U_2325_x.size - 8:]
xy_for_interpol = x_for_interpol * y_for_interpol
xx_for_interpol = x_for_interpol * x_for_interpol
yy_for_interpol = y_for_interpol * y_for_interpol

k_2325 = (np.mean(xy_for_interpol) - np.mean(x_for_interpol) * np.mean(y_for_interpol)) / (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2)
k_err_2325 = np.sqrt(1 / 10 * np.fabs((np.mean(yy_for_interpol) - np.mean(y_for_interpol) ** 2) / \
        (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2) - k_2325 ** 2))
b_2325 = np.mean(y_for_interpol) - k_2325 * np.mean(x_for_interpol)
b_err_2325 = k_err_2325 * np.sqrt(np.mean(xx_for_interpol))

print("For 2325:") 
print("\tk:", k_2325, "+-", k_err_2325)
print("\tb:", b_2325, "+-", b_err_2325)

V_0_2325 = b_2325 / k_2325
V_0_err_2325 = V_0_2325 * np.sqrt(k_err_2325 ** 2 / k_2325 ** 2 + b_err_2325 ** 2 / b_2325 ** 2)

print("Blocked for 2325:", V_0_2325, "+-", V_0_err_2325)

# main

V_0_arr = np.array([V_0_2200, V_0_2325, V_0_2550])
V_0_err_arr = np.array([V_0_err_2200, V_0_err_2325, V_0_err_2550])
freq_arr = np.array([freq_2200, freq_2325, freq_2550])
freq_err_arr = np.array([0, 0, 0])

x_for_interpol  = freq_arr
y_for_interpol  = V_0_arr
xy_for_interpol = x_for_interpol * y_for_interpol
xx_for_interpol = x_for_interpol * x_for_interpol
yy_for_interpol = y_for_interpol * y_for_interpol

k_main = (np.mean(xy_for_interpol) - np.mean(x_for_interpol) * np.mean(y_for_interpol)) / (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2)
k_main_err = np.sqrt(1 / 10 * np.fabs((np.mean(yy_for_interpol) - np.mean(y_for_interpol) ** 2) / \
        (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2) - k_main ** 2))
b_main = np.mean(y_for_interpol) - k_main * np.mean(x_for_interpol)
b_main_err = k_main_err * np.sqrt(np.mean(xx_for_interpol))

print("h is", k_main * 1.6 * 1e-19, "+-", k_main_err * 1.6 * 1e-19)

points_x = np.linspace(freq_arr[0], freq_arr[2], 1000)
points_y = points_x * k_main + b_main

plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linewidth = 2)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.scatter(points_x, points_y)
plt.errorbar(freq_arr, V_0_arr, xerr = freq_err_arr, yerr = V_0_err_arr, fmt = 'o:w', ecolor = 'red')
plt.show()