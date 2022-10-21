from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import math as mth
from scipy.interpolate import interp1d

# p - x
# I - y

pdf = PdfPages("Figures.pdf")

room_pressure = 737

#arr_fisrt = [14, 14.5, 14.5, 15, 16.5, 17, 15.5, 16, 16.5, 18, 19, 19.5]
#print(np.mean(arr_fisrt))
#print(np.std(arr_fisrt, ddof = 1))
#arr_err_first = [np.std(arr_fisrt), 0.7]
#print("err first: %f", mth.sqrt(arr_err_first[0]**2 + arr_err_first[1]**2))

# 3

arr_part_3_linear = np.loadtxt('part_3_linear.txt', dtype = float)
x_part_3_linear = room_pressure - arr_part_3_linear[0]
y_part_3_linear = arr_part_3_linear[1]
x_part_3_linear_err = 5
y_part_3_linear_err = 3

arr_part_3_plato = np.loadtxt('part_3_plato.txt', dtype = float)
x_part_3_plato = room_pressure - arr_part_3_plato[0]
y_part_3_plato = arr_part_3_plato[1]
x_part_3_plato_err = 5
y_part_3_plato_err = 3

plt.figure()
plt.minorticks_on()
plt.grid(which='major',
         color = 'k', 
         linewidth = 2)
plt.grid(which='minor', 
         color = 'k', 
         linestyle = ':')
plt.xlabel("p, торр")
plt.ylabel("I, pA")
plt.title("Зависимость тока от давления в камере")
plt.errorbar(x_part_3_linear, y_part_3_linear, xerr = x_part_3_linear_err, yerr = y_part_3_linear_err, fmt='o-r')
plt.errorbar(x_part_3_plato, y_part_3_plato, xerr = x_part_3_plato_err, yerr = y_part_3_plato_err, fmt='o-r')
# plt.errorbar(x_8, y_8, xerr=x_8_err, yerr=y_8_err, fmt='o-g', ecolor='green')
plt.show()

r_l = 5
T = 292
T_err = 1
p = 560
p_err = 10
T_norm = 288
p_norm = 760

r_norm = r_l * p / p_norm * T_norm / T;
r_norm_err = r_norm * np.sqrt((p_err / p)**2 + (T_err / T)**2)
print('r: ', r_norm, ' +- ', r_norm_err)
E = (r_norm / 0.32) ** (2 / 3)
E_err = 2 / 3 / ((0.32) ** (2 / 3)) * (r_norm ** (-1 / 3)) * r_norm_err
print('E: ', E, ' +- ', E_err)

# 2

arr_part_2 = np.loadtxt('part_2.txt', dtype = float)
x_part_2 = room_pressure - arr_part_2[0]
y_part_2 = arr_part_2[1]
x_part_2_err = 5
y_part_2_err = 3

p_middle = 125
p_middle_err = 10

x_for_interpol = np.empty(10)
y_for_interpol = np.empty(10)
for i in range(6, 16):
        x_for_interpol[i - 6] = x_part_2[i]
        y_for_interpol[i - 6] = y_part_2[i]
print(x_for_interpol)
print(y_for_interpol)

interpol_func = interp1d(y_for_interpol, x_for_interpol, kind = 'linear', fill_value = "extrapolate")
p_extr = interpol_func(0)
p_line = interpol_func(2000)
print(p_extr)

# compute k and sigma_k:

xy_for_interpol = x_for_interpol * y_for_interpol
xx_for_interpol = x_for_interpol * x_for_interpol
yy_for_interpol = y_for_interpol * y_for_interpol

k = (np.mean(xy_for_interpol) - np.mean(x_for_interpol) * np.mean(y_for_interpol)) / (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2)
k_err = np.sqrt(1 / 8 * np.fabs((np.mean(yy_for_interpol) - np.mean(y_for_interpol) ** 2) / (np.mean(xx_for_interpol) - np.mean(x_for_interpol) ** 2) - k ** 2))

print("k: ", k, " +- ", k_err)

line_arr_x = [p_line, p_extr]
line_arr_y = [2000, 0]

p_2 = 225
p_2_err = 13
r_l_2 = 9

r_norm_2 = r_l_2 * p_2 / p_norm * T_norm / T;
r_norm_2_err = r_norm_2 * np.sqrt((p_2_err / p_2)**2 + (T_err / T)**2)
print('r_2: ', r_norm_2, ' +- ', r_norm_2_err)
E_2 = (r_norm_2 / 0.32) ** (2 / 3)
E_2_err = 2 / 3 / ((0.32) ** (2 / 3)) * (r_norm_2 ** (-1 / 3)) * r_norm_2_err
print('E: ', E_2, ' +- ', E_2_err)

plt.figure()
plt.minorticks_on()
plt.grid(which='major',
         color = 'k', 
         linewidth = 2)
plt.grid(which='minor', 
         color = 'k', 
         linestyle = ':')
plt.xlabel("p, торр")
plt.ylabel("N")
plt.title("Количество частиц в зависимости от давления")
plt.errorbar(x_part_2, y_part_2, xerr = x_part_2_err, yerr = y_part_2_err, fmt='o-r')
plt.plot(line_arr_x, line_arr_y, '-r')
plt.show()

#pdf.savefig()
pdf.close()