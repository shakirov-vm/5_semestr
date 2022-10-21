# 1
import numpy as np

# 2
arr_1 = np.array([1, 3, 5, 7, 9])
arr_2 = np.arange(start = 1, stop = 11, step = 2)
arr_3 = np.linspace(start = 1, stop = 9, num = 5, dtype = int)

# 3
arr_3_100 = np.linspace(start = 1, stop = 100, num = 100, dtype = int)

# 4
x_k = np.empty(100, dtype = float)
for i in range(100):
  x_k[i] = np.sin(2 * np.pi * 0.07 * i)

# 5
import matplotlib.pyplot as plt

# 6
plt.figure(figsize=[12, 5], dpi=100)

# 7
plt.plot(arr_3_100, x_k)

# 8
plt.stem(arr_3_100, x_k, '--r', use_line_collection = True)

plt.show()
# 9
def sin_x_div_x(x):
  if x == 0:
    return 100
  else: 
    return np.sin(x)/x

arr_new = np.empty(100, dtype = float)
for i in range(100):
  arr_new[i] = sin_x_div_x(arr_3_100[i])

plt.plot(arr_3_100, arr_new)
plt.title("sinx / x")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# 10
arr_z = np.empty(100, dtype = "complex_")
for i in range(100):
  arr_z[i] = np.exp(-1 * 1j * 2 * np.pi * 0.07 * i)

z_real = np.empty(100, dtype = "complex_")
for i in range(100):
  z_real[i] = arr_z[i].real

plt.plot(arr_3_100, z_real)
plt.show()

z_imag = np.empty(100, dtype = "complex_")
for i in range(100):
  z_imag[i] = arr_z[i].imag

plt.plot(arr_3_100, z_imag)
plt.show()
