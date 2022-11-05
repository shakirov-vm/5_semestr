from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import math as mth
from scipy.interpolate import interp1d

arr_T_AChT = np.loadtxt('AChT.txt', dtype = float)

print("AChT mean:", np.mean(arr_T_AChT), "+-", np.std(arr_T_AChT, ddof = 1))

arr_SB = np.loadtxt('SB.txt', dtype = float)

arr_T = arr_SB[0]
arr_I = arr_SB[1]
arr_U = arr_SB[2]

arr_W = arr_I * arr_U
print(arr_W)

arr_I_err = np.linspace(0.05, 0.05, 9)
arr_U_err = np.linspace(0.5, 0.5, 9)
arr_W_err = np.zeros(9)

for i in range(9):
	arr_W_err[i] = np.sqrt((arr_I_err[i] / arr_I[i])**2 + (arr_U_err[i] / arr_U[i])**2)

arr_T_err = np.zeros(9)

for i in range(9):
	arr_T_err[i] = arr_T[i] * 0.1
	arr_T[i] += 273

arr_ln_T = np.zeros(9)
arr_ln_W = np.zeros(9)
arr_ln_T_err = np.zeros(9)
arr_ln_W_err = np.zeros(9)

for i in range(9):
	arr_ln_T[i] = np.log(arr_T[i])
	arr_ln_T_err[i] = 1 / arr_T[i] * arr_T_err[i]
	arr_ln_W[i] = np.log(arr_W[i])
	arr_ln_W_err[i] = 1 / arr_W[i] * arr_W_err[i]

plt.figure()
plt.minorticks_on()
plt.xlabel("ln T")
plt.ylabel("ln W")
plt.title("Зависимость мощности излучения от температуры (log)")
plt.grid(which='major',
        color = 'k', 
        linewidth = 2)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.errorbar(arr_ln_T, arr_ln_W, xerr = arr_ln_T_err, yerr = arr_ln_W_err, fmt = 'o-r', ecolor = 'red')
plt.show()		# ??

x_for_interpol = arr_ln_T
y_for_interpol = arr_ln_W

# compute k and sigma_k:

xy_for_interpol = x_for_interpol * y_for_interpol
xx_for_interpol = x_for_interpol * x_for_interpol
yy_for_interpol = y_for_interpol * y_for_interpol

print(arr_T)
print(arr_W)

k = (np.mean(xy_for_interpol) - np.mean(x_for_interpol) * np.mean(y_for_interpol)) / (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2)
k_err_rand = np.sqrt(1 / 7 * np.fabs((np.mean(yy_for_interpol) - np.mean(y_for_interpol) ** 2) / (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2) - k ** 2))
k_min = 1 / (arr_ln_T[8] + arr_ln_T_err[8] - (arr_ln_T[0] - arr_ln_T_err[0])) * (arr_ln_W[8] - arr_ln_W_err[8] - (arr_ln_W[0] + arr_ln_W_err[0]))
k_max = 1 / (arr_ln_T[8] - arr_ln_T_err[8] - (arr_ln_T[0] + arr_ln_T_err[0])) * (arr_ln_W[8] + arr_ln_W_err[8] - (arr_ln_W[0] - arr_ln_W_err[0]))
k_err_sys = (k_max - k) / 2 + (k - k_min) / 2
k_err = np.sqrt(k_err_rand**2 + k_err_sys**2)
print("err:", k_min, k, k_max, "rand:", k_err_rand, "syst:", k_err_sys)
print("k:", k, "+-", k_err, "\n")
print("T:", arr_T[6], "+-", arr_T_err[6], "W:", arr_W[6], "+-", arr_W_err[6])
sigma = arr_W[6] / 0.232 / 0.36 / (arr_T[6]**4)
sigma_err = sigma * np.sqrt((arr_W_err[6] / arr_W[6])**2 + (4 * arr_T_err[6] / arr_T[6]) ** 2)
print("sigma:", sigma, "+-", sigma_err)
sigma /= 1000
sigma_err /= 1000
h = (2 * np.pi**5 * 1.38e-23**4 / 15 / 3e10**2 / sigma) ** (1/3)
h_err = 1/3 * sigma_err / sigma * h
print("h:", h, "+-", h_err)