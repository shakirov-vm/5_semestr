from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import math as mth
from scipy.interpolate import UnivariateSpline
from scipy import interpolate

pdf = PdfPages("Figures.pdf")

calibration = np.loadtxt('calibration.txt', dtype = float)

calibration_x = calibration[0]
calibration_y = calibration[1]

calibration_x_err = np.linspace(5, 5, 29)

plt.figure()
plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linestyle = ':')
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
# plt.errorbar(calibration_x, calibration_y, xerr = calibration_x_err, fmt = 'o-k', ecolor = 'black')
plt.axvline(x = 2480, color = 'r')
plt.axvline(x = 1486, color = 'b')
plt.axvline(x = 841, color = 'm')
plt.axvline(x = 425, color = 'm')

plt.axvline(x = 2286, color = 'c')
plt.axvline(x = 2181, color = 'c')
plt.axvline(x = 1796, color = 'c')

calibration_x.sort()
calibration_y.sort()

print(calibration_x)

spl = UnivariateSpline(calibration_x, calibration_y)
spl_x = np.linspace(300, 2700, 10000)
plt.plot(spl_x, spl(spl_x), 'r', linewidth = 2)
plt.scatter(calibration_x, calibration_y, color = 'r', s = 20)
# theta = np.polyfit(calibration_x, calibration_y, 2)
# y_line = theta[2] + theta[1] * pow(calibration_x, 1) + theta[0] * pow(calibration_x, 2)
# plt.plot(calibration_x, y_line, 'o-k')

# plt.scatter(calibration_x, calibration_y, fmt = 'o-r', ecolor = 'red')
plt.show()

H_x = np.array([0.139, 0.188, 0.21, 0.222])
H_y = np.array([1.53, 2.06, 2.31, 2.44])

f = interpolate.interp1d(H_x, H_y)

plt.scatter(H_x, H_y, color = 'r', s = 20)

print("f(0.15) =", f(0.15), ", f(0.2) =", f(0.2))

plt.show()

# pdf.savefig()
# pdf.close()