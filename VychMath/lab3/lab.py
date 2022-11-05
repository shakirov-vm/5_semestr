import numpy as np

def f(x):
	return x**2 - np.exp(x) / 5
def dev_f(x):
	return np.exp(x) / 5

def newton_s_method(func, dev_func, x_0, eps):

	discrepancy = 1000000
	i = 0
# no do-while in python?
	while discrepancy > eps:
		#  		 ???
		x_0 = x_0 + func(x_0) / dev_func(x_0)
		discrepancy = np.abs(-func(x_0))
		i += 1
	print("Newton say: for", i, "iterations x:", x_0, "discrepancy:", discrepancy)
	return x_0

def simple_iterations_method(func, phi, x_0, eps):
	discrepancy = 1000000
	i = 0

	while discrepancy > eps:
		x_0 = phi(x_0)
		discrepancy = np.abs(-func(x_0))
		i += 1

	print("SIM: for", i, "iterations x:", x_0, "discrepancy:", discrepancy)
	return x_0

def simple_iterations_method_custom(func, tau, x_0, eps):
	discrepancy = 1000000
	i = 0

	while discrepancy > eps:
		x_0 = x + tau * func(x_0)
		discrepancy = np.abs(-func(x_0))
		i += 1

	print("SIM Custom: for", i, "iterations x:", x_0, ", discrepancy:", discrepancy)
	return x_0

print("For non-linear equation solutions obtained from simple iterations method:")
# x = newton_s_method(f, dev_f, 5, 10e-6)
x = simple_iterations_method(f, lambda x: np.exp(x / 2) / (5 ** (1/2)), 0, 10e-6)
x = simple_iterations_method(f, lambda x: np.log(5 * x**2), 4, 10e-6)
x = simple_iterations_method(f, lambda x: -np.exp(x / 2) / (5 ** (1/2)), 3, 10e-6)
# x = simple_iterations_method_custom(f, 0.2, -2, 10e-6)


# 2x^2 - xy - 5x + 1 = 0
# x + 3 lgx - y^2 = 0

def norm(x, y): # vector 2D
	return np.sqrt(x ** 2 + y ** 2)

def newton_s_method_system(F, I, x_0, eps):
	discrepancy = 1000000
	i = 0

	while discrepancy > eps: # maybe #define?
		x_0[0] = x_0[0] - I[0][0](x_0[0], x_0[1]) * F[0](x_0[0], x_0[1]) - I[0][1](x_0[0], x_0[1]) * F[1](x_0[0], x_0[1])
		x_0[1] = x_0[1] - I[1][0](x_0[0], x_0[1]) * F[0](x_0[0], x_0[1]) - I[1][1](x_0[0], x_0[1]) * F[1](x_0[0], x_0[1])

		discrepancy = norm(F[0](x_0[0], x_0[1]), F[1](x_0[0], x_0[1]))
		i += 1
	print("For non-linear system of equation solutions obtained from newton's method:")
	print("\tfor", i, "iterations ( x , y ): (", x_0[0], ",", x_0[1], "), discrepancy: ", discrepancy)
	return x_0[0], x_0[1]

F1 = lambda x, y: np.sin(x) - y - 1.32
F2 = lambda x, y: np.cos(y) - x + 0.85

# This is inverse jacobian:

I11 = lambda x, y: np.sin(y) / (np.cos(x) * np.sin(y) + 1)
I21 = lambda x, y: -1 / (np.cos(x) * np.sin(y) + 1)
I12 = lambda x, y: -1 / (np.cos(x) * np.sin(y) + 1)
I22 = lambda x, y: -np.cos(x) / (np.cos(x) * np.sin(y) + 1)

F_vec = np.array([F1, F2])
I_matr = np.array([[I11, I12], [I21, I22]])
x_0_vec = np.array([1.5, -0.7])

newton_s_method_system(F_vec, I_matr, x_0_vec, 10e-6)

