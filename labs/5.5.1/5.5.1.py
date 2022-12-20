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

n_phon = 2511

ferr = np.loadtxt('Ferr.txt', dtype = float)
alum = np.loadtxt('Alum.txt', dtype = float)
plum = np.loadtxt('Plum.txt', dtype = float)
cork = np.loadtxt('Cork.txt', dtype = float)

n_0_arr = np.array([ ferr[1][0] - n_phon, alum[1][0] - n_phon, plum[1][0] - n_phon, cork[1][0] - n_phon ])

n_0 = np.mean(n_0_arr)
n_0_err = np.sqrt(np.var(n_0_arr))

print("n_0:", n_0, "+-", n_0_err)

#in cm

ferr_l = ferr[0]
alum_l = alum[0] * 2
plum_l = plum[0] * 0.5
cork_l = cork[0] * 2

ferr_N = ferr[1] - n_phon
alum_N = alum[1] - n_phon
plum_N = plum[1] - n_phon
cork_N = cork[1] - n_phon

ferr_N_err = ferr_N * n_0_err / n_0
alum_N_err = alum_N * n_0_err / n_0
plum_N_err = plum_N * n_0_err / n_0
cork_N_err = cork_N * n_0_err / n_0

ln_ferr_y = np.log(n_0 / ferr_N)
ln_alum_y = np.log(n_0 / alum_N)
ln_plum_y = np.log(n_0 / plum_N)
ln_cork_y = np.log(n_0 / cork_N)

ferr_l_err = ferr_l * 0.05
alum_l_err = alum_l * 0.05
plum_l_err = plum_l * 0.05
cork_l_err = cork_l * 0.05

ln_ferr_y_err = ln_ferr_y * ferr_N / n_0 * np.sqrt((n_0_err / n_0) ** 2 + (n_0 / ferr_N / ferr_N * ferr_N_err) ** 2)
ln_alum_y_err = np.abs(ln_alum_y) * alum_N / n_0 * np.sqrt((n_0_err / n_0) ** 2 + (n_0 / alum_N / alum_N * alum_N_err) ** 2)
ln_plum_y_err = np.abs(ln_plum_y) * plum_N / n_0 * np.sqrt((n_0_err / n_0) ** 2 + (n_0 / plum_N / plum_N * plum_N_err) ** 2)
ln_cork_y_err = np.abs(ln_cork_y) * cork_N / n_0 * np.sqrt((n_0_err / n_0) ** 2 + (n_0 / cork_N / cork_N * cork_N_err) ** 2)

k_ferr, k_err_ferr, b_ferr, b_err_ferr = MNK(ferr_l, ln_ferr_y)
k_alum, k_err_alum, b_alum, b_err_alum = MNK(alum_l, ln_alum_y)
k_plum, k_err_plum, b_plum, b_err_plum = MNK(plum_l, ln_plum_y)
k_cork, k_err_cork, b_cork, b_err_cork = MNK(cork_l, ln_cork_y)

points_ferr_x = np.linspace(ferr_l[0], ferr_l[ferr_l.size - 1], 1000)
points_ferr_y = points_ferr_x * k_ferr + b_ferr

points_alum_x = np.linspace(alum_l[0], alum_l[alum_l.size - 1], 1000)
points_alum_y = points_alum_x * k_alum + b_alum

points_plum_x = np.linspace(plum_l[0], plum_l[plum_l.size - 1], 1000)
points_plum_y = points_plum_x * k_plum + b_plum

points_cork_x = np.linspace(cork_l[0], cork_l[cork_l.size - 1], 1000)
points_cork_y = points_cork_x * k_cork + b_cork

plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linewidth = 1)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.title("Ferrum")
plt.xlabel("l, cm")
plt.ylabel("ln N_0 / N")
plt.errorbar(ferr_l, ln_ferr_y, xerr = ferr_l_err, yerr = ln_ferr_y_err, fmt = 'or')
plt.plot(points_ferr_x, points_ferr_y, 'black', linewidth = 2)
plt.show()

plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linewidth = 1)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.title("Aluminium")
plt.xlabel("l, cm")
plt.ylabel("ln N_0 / N")
plt.errorbar(alum_l, ln_alum_y, xerr = alum_l_err, yerr = ln_alum_y_err, fmt = 'or')
plt.plot(points_alum_x, points_alum_y, 'black', linewidth = 2)
plt.show()

plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linewidth = 1)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.title("Plumbum")
plt.xlabel("l, cm")
plt.ylabel("ln N_0 / N")
plt.errorbar(plum_l, ln_plum_y, xerr = plum_l_err, yerr = ln_plum_y_err, fmt = 'or')
plt.plot(points_plum_x, points_plum_y, 'black', linewidth = 2)
plt.show()

plt.minorticks_on()
plt.grid(which='major',
        color = 'k', 
        linewidth = 1)
plt.grid(which='minor', 
        color = 'k', 
        linestyle = ':')
plt.title("Cork")
plt.xlabel("l, cm")
plt.ylabel("ln N_0 / N")
plt.errorbar(cork_l, ln_cork_y, xerr = cork_l_err, yerr = ln_cork_y_err, fmt = 'or')
plt.plot(points_cork_x, points_cork_y, 'black', linewidth = 2)
plt.show()

mu_ferr, mu_ferr_err, a, a = MNK(ferr_l, ln_ferr_y)
mu_alum, mu_alum_err, a, a = MNK(alum_l, ln_alum_y)
mu_plum, mu_plum_err, a, a = MNK(plum_l, ln_plum_y)
mu_cork, mu_cork_err, a, a = MNK(cork_l, ln_cork_y)

print("ferr:", mu_ferr, "+-", mu_ferr_err)
print("alum:", mu_alum, "+-", mu_alum_err)
print("plum:", mu_plum, "+-", mu_plum_err)
print("cork:", mu_cork, "+-", mu_cork_err)