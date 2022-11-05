from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import math as mth

# V - x
# I - y

pdf = PdfPages("Figures.pdf")

arr_fisrt = [14, 14.5, 14.5, 15, 16.5, 17, 15.5, 16, 16.5, 18, 19, 19.5]
print(np.mean(arr_fisrt))
print(np.std(arr_fisrt, ddof = 1))
arr_err_first = [np.std(arr_fisrt), 0.7]
print("err first: %f", mth.sqrt(arr_err_first[0]**2 + arr_err_first[1]**2))

arr_4 = np.loadtxt('data_4_format.txt', dtype = float)
print(arr_4)
x_4 = arr_4[0]
y_4 = arr_4[1]
x_4_err = arr_4[0] * 0.0003 + 0.04
y_4_err = arr_4[1] * 0.005  + 0.0001

arr_6 = np.loadtxt('data_6_format.txt', dtype = float)
print(arr_6)
x_6 = arr_6[0]
y_6 = arr_6[1]
x_6_err = arr_6[0] * 0.0003 + 0.04
y_6_err = arr_6[1] * 0.005  + 0.0001

arr_8 = np.loadtxt('data_8_format.txt', dtype = float)
print(arr_8)
x_8 = arr_8[0]
y_8 = arr_8[1]
x_8_err = arr_8[0] * 0.0003 + 0.04
y_8_err = arr_8[1] * 0.005  + 0.0001


plt.figure()
plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linewidth = 2)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.errorbar(x_4, y_4, xerr=x_4_err, yerr=y_4_err, fmt='o-r', ecolor='red')
plt.errorbar(x_6, y_6, xerr=x_6_err, yerr=y_6_err, fmt='o-b', ecolor='blue')
plt.errorbar(x_8, y_8, xerr=x_8_err, yerr=y_8_err, fmt='o-g', ecolor='green')
plt.show()

pdf.savefig()
pdf.close()