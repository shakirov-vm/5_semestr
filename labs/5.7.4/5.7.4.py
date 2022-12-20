import numpy as np
import matplotlib.pyplot as plt
import math as mth
from scipy.interpolate import UnivariateSpline

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

data = np.loadtxt('data.txt', dtype = float)

thickness_x = data[0]
N_y = data[1]

k, k_err, b, b_err = MNK(thickness_x, N_y)

points_x = np.linspace(thickness_x[0], thickness_x[thickness_x.size - 1], 1000)
points_y = points_x * k + b

plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linewidth = 2)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.title("Зависимость количества детектированных частиц от толщины пластин")
plt.xlabel("d, см")
plt.ylabel("N")
plt.errorbar(thickness_x, N_y, fmt = 'ok')
plt.scatter(points_x, points_y, linewidth = 1)
plt.show()

N_y_ligth = N_y[0] - points_y[999]
N_y_ligth_err = N_y_ligth * np.sqrt(k_err * k_err / k / k + b_err * b_err / b / b)
print(N_y_ligth, N_y_ligth_err)