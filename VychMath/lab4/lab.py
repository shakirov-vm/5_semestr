import numpy as np
import matplotlib.pyplot as plt

years = np.arange(1910, 2010, 10)
quantity = np.array([92228496, 106021537, 123202624, 132164569, 151325798, 179323175, 203211926, 226545805, 248709873, 281421906])

# Форма Ньютона:
# slow and don't need?
def get_newton_coef(x, f_x, n, start, stop):

	if stop - start != n:
		print("Newton Warning\n")

	if n == 1:
		return (f_x[stop] - f_x[start]) / (x[stop] - x[start])
	else:
		return (get_newton_coef(x, f_x, n - 1, start + 1, stop) - get_newton_coef(x, f_x, n - 1, start, stop - 1)) / (x[stop] - x[start])

def get_newton_polynom(x, f_x):

	coeffs = np.zeros(x.size)

	coeffs[0] = f_x[0]
	for i in range(1, x.size):
		coeffs[i] = get_newton_coef(x, f_x, i, 0, i)

	# print(coeffs)
	return coeffs

def get_newton_result(x, f_x, x_0):

	result = 0
	mul = 1

	coeffs = get_newton_polynom(x, f_x)
	for i in range(x.size):
		result += coeffs[i] * mul
		mul *= (x_0 - x[i])
	return result

res = get_newton_result(years, quantity, 2010)
print("Newton's polynom say, that in USA will", res, "people in 2010")

points = np.arange(1890, 2011, 1)
y = np.zeros(points.size)

for i in range(points.size):
	y[i] = get_newton_result(years, quantity, points[i])

#plt.plot(points, y)

#plt.title('Ньютон')
#plt.xlabel('Год')
#plt.ylabel('Население')

#plt.grid()
#plt.show()

# Spline

h = 10
num_spl = years.size - 1

arr_c = np.zeros(num_spl + 1)

system_size = num_spl - 1

# Прямая прогонка:

p = np.zeros(system_size + 1)
r = np.zeros(system_size + 1)

for i in range(1, num_spl):

	F_next = ((quantity[i + 1] - quantity[i]) / h - (quantity[i] - quantity[i - 1]) / h)
	p[i] = - h / (h * p[i - 1] + 4 * h)
	r[i] = (F_next - h * r[i - 1]) / (h * p[i - 1] + 4 * h)

arr_c[num_spl] = 0
arr_c[num_spl - 1] = r[system_size]

for i in range(num_spl - 1, 0, -1):
      arr_c[i] = p[i] * arr_c[i + 1] + r[i]

arr_a = np.zeros(num_spl + 1)
arr_b = np.zeros(num_spl + 1)
arr_d = np.zeros(num_spl + 1)

arr_a[0] = quantity[0]

for i in range(num_spl, 0, -1):
	arr_d[i] = 1 / h / 3 * (arr_c[i] - arr_c[i - 1])
	arr_b[i] = h * (arr_c[i] / 3 + arr_c[i - 1] / 6) + (quantity[i] - quantity[i - 1]) / h
	arr_a[i] = quantity[i]

def calc_from_spline(x_0, x_k, a, b, c, d):
	return a + b * (x_0 - x_k) + c * (x_0 - x_k) * (x_0 - x_k) + d * (x_0 - x_k) * (x_0 - x_k) * (x_0 - x_k)

res = calc_from_spline(2010, 2000, arr_a[9], arr_b[9], arr_c[9], arr_d[9])

print("Spline's polynom say, that in USA will", res, "people in 2010")
print("In 2010 there were 308,745,538 people in the USA")