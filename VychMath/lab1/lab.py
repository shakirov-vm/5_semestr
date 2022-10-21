import numpy as np
import matplotlib.pyplot as plt
import math

width = 20
x = 1.4

analitics_1 = 2 * x * math.cos(x**2)
analitics_2 = -1 * math.cos(x) * math.sin(math.sin(x))
analitics_3 = -1 * math.exp(math.sin(math.cos(x))) * math.sin(x) * math.cos(math.cos(x))
analitics_4 = 1 / (x + 3)
analitics_5 = 1 / (2 * ((x + 3)**0.5))

def first_method(x, h, func):
	
	return (func(x + h) - func(x)) / h

def secnd_method(x, h, func):

	return (func(x) - func(x - h)) / h

def third_method(x, h, func):

	return (func(x + h) - func(x - h)) / (2 * h)

def forth_method(x, h, func):

	return ((4 / 3) * ((func(x + h) - func(x - h)) / (2 * h))) - ((1 / 3) * ((func(x + 2 * h) - func(x - 2 * h)) / (4 * h)))

def fifth_method(x, h, func):
	
	return ((3 / 2) * ((func(x + h) - func(x - h)) / (2 * h))) - ((3 / 5) * ((func(x + 2 * h) - func(x - 2 * h)) / (4 * h))) + ((1 / 10) * ((func(x + 3 * h) - func(x - 3 * h)) / (6 * h)))

def show_method_graph(method): # legacy

	x_arr   = np.empty([width], dtype = float)
	y_arr_1 = np.empty([width], dtype = float)
	y_arr_2 = np.empty([width], dtype = float)
	y_arr_3 = np.empty([width], dtype = float)
	y_arr_4 = np.empty([width], dtype = float)
	y_arr_5 = np.empty([width], dtype = float)
	
	analitics_1 = 2 * x * math.cos(x**2)
	analitics_2 = -1 * math.cos(x) * math.sin(math.sin(x))
	analitics_3 = -1 * math.exp(math.sin(math.cos(x))) * math.sin(x) * math.cos(math.cos(x))
	analitics_4 = 1 / (x + 3)
	analitics_5 = 1 / (2 * ((x + 3)**0.5))

	for i in range(0, width):

		x_arr[i] = 1 / (2**i)
		y_arr_1[i] = math.fabs(method(x, x_arr[i], lambda x : math.sin(x**2)) - analitics_1)
		y_arr_2[i] = math.fabs(method(x, x_arr[i], lambda x : math.cos(math.sin(x))) - analitics_2)
		y_arr_3[i] = math.fabs(method(x, x_arr[i], lambda x : math.exp(math.sin(math.cos(x)))) - analitics_3)
		y_arr_4[i] = math.fabs(method(x, x_arr[i], lambda x : math.log(x + 3)) - analitics_4)
		y_arr_5[i] = math.fabs(method(x, x_arr[i], lambda x : (x + 3)**0.5) - analitics_5)

	plt.figure()
	plt.minorticks_on()
	plt.grid(which='major', color = 'k', linewidth = 2)
	plt.grid(which='minor', color = 'k', linestyle = ':')
	plt.semilogx()
	plt.semilogy()
	plt.errorbar(x_arr, y_arr_1, fmt='o-c')
	plt.errorbar(x_arr, y_arr_2, fmt='o-b')
	plt.errorbar(x_arr, y_arr_3, fmt='o-g')
	plt.errorbar(x_arr, y_arr_4, fmt='o-r')
	plt.errorbar(x_arr, y_arr_5, fmt='o-m')
	plt.show()

def show_function_graph(func, analitics, str_title):

	x_arr   = np.empty([width], dtype = float)
	y_arr_1 = np.empty([width], dtype = float)
	y_arr_2 = np.empty([width], dtype = float)
	y_arr_3 = np.empty([width], dtype = float)
	y_arr_4 = np.empty([width], dtype = float)
	y_arr_5 = np.empty([width], dtype = float)
	

	for i in range(0, width):

		x_arr[i] = 1 / (2**i)
		y_arr_1[i] = math.fabs(first_method(x, x_arr[i], func) - analitics)
		y_arr_2[i] = math.fabs(secnd_method(x, x_arr[i], func) - analitics)
		y_arr_3[i] = math.fabs(third_method(x, x_arr[i], func) - analitics)
		y_arr_4[i] = math.fabs(forth_method(x, x_arr[i], func) - analitics)
		y_arr_5[i] = math.fabs(fifth_method(x, x_arr[i], func) - analitics)

	plt.figure()
	plt.minorticks_on()
	plt.grid(which='major', color = 'k', linewidth = 2)
	plt.grid(which='minor', color = 'k', linestyle = ':')
	plt.semilogx()
	plt.semilogy()
	plt.title(str_title)
	plt.errorbar(x_arr, y_arr_1, fmt='o-c')
	plt.errorbar(x_arr, y_arr_2, fmt='o-b')
	plt.errorbar(x_arr, y_arr_3, fmt='o-g')
	plt.errorbar(x_arr, y_arr_4, fmt='o-r')
	plt.errorbar(x_arr, y_arr_5, fmt='o-m')
	plt.show()

#start:

show_function_graph(lambda x : math.sin(x**2), analitics_1, "sin(x^2)")
show_function_graph(lambda x : math.cos(math.sin(x)), analitics_2, "cos(sin(x))")
show_function_graph(lambda x : math.exp(math.sin(math.cos(x))), analitics_3, "exp(sin(cos(x)))")
show_function_graph(lambda x : math.log(x + 3), analitics_4, "ln(x + 3)")
show_function_graph(lambda x : (x + 3)**0.5, analitics_5, "(x + 3)^0.5")
