import numpy as np

n = 100
eps = 0.000001

def create_matrix():
	matrix = np.zeros((100, 100))

	for i in range(100):
		matrix[0][i] = 1
	for i in range(1, 99):
		matrix[i][i] = 10
		matrix[i][i - 1] = 1
		matrix[i][i + 1] = 1
	matrix[99][98] = 1
	matrix[99][99] = 1

	return matrix

def create_f_vector():
	vector = np.empty(100)

	for i in range(100):
		vector[i] = 100 - i

	return vector

def check_LU(matrix):

	for i in range(n):
		det = np.linalg.det(matrix[0 : i, 0 : i])
		if det == 0:
			print("LU isn't exist")
			return False
	print("LU exist")
	return True

def LU_factorization(matrix):

	L = np.zeros((n, n))
	U = np.zeros((n, n))

	for i in range(n):
		for k in range(i, n):
			sum = 0;
			for j in range(i):
				sum += L[i][j] * U[j][k]
			U[i][k] = matrix[i][k] - sum
		for k in range(i, n):
			if i == k:
				L[i][i] = 1
			else:
				sum = 0
				for j in range(i):
					sum += L[k][j] * U[j][i]
				L[k][i] = (matrix[k][i] - sum) / U[i][i]
	return L, U


def get_vec_norm(vec):
	return np.sqrt(np.dot(vec, vec)) # 3-rd norm
	#return np.linalg.norm(vec)

def get_max_lambda(matrix): # absolute value

	y_k = np.zeros(n)
	y_k[1] = 0.2
	y_k[3] = 0.3
	y_k_plus_1 = y_k

	lambda_1 = 1
	lambda_2 = 0

	epsilon = 0.01

	while np.abs(lambda_2 - lambda_1) > epsilon:

		y_k = y_k_plus_1
		y_k_plus_1 = matrix.dot(y_k)
		lambda_2 = lambda_1
		lambda_1 = np.dot(y_k_plus_1, y_k) / np.dot(y_k, y_k) # 3-rd norm

	return lambda_1

def get_min_lambda(matrix): # absolute value

	matrix = np.linalg.inv(matrix)

	y_k = np.zeros(n)
	y_k[1] = 0.2
	y_k[3] = 0.3
	y_k_plus_1 = y_k

	lambda_1 = 1
	lambda_2 = 0

	epsilon = 0.01

	while np.abs(lambda_2 - lambda_1) > epsilon:

		y_k = y_k_plus_1
		y_k_plus_1 = matrix.dot(y_k)
		lambda_2 = lambda_1
		lambda_1 = np.dot(y_k_plus_1, y_k) / np.dot(y_k, y_k) # 3-rd norm

	return lambda_1

def compute_matrix_condition_num(matrix): # 3-rd norm

	matrix_inv = np.linalg.inv(matrix)

	A_norm = np.sqrt(get_max_lambda((matrix.conj().T).dot(matrix)))
	print(matrix)
	A_inv_norm = np.sqrt(get_max_lambda((matrix_inv.conj().T).dot(matrix_inv)))

	MCN = A_norm / A_inv_norm
	print("matrix condition number:", MCN)

def solve_LU(L, U, f):

	v = np.zeros(n)
	
	for i in range(n):
		sum = 0
		if i != 0:	
			for j in range(i):
				sum += L[i][j] * v[j]
		v[i] = f[i] - sum

	u = np.zeros(n)

	for i in range(n - 1, -1, -1):
		sum = 0
		for j in range(i + 1, n):
			sum += U[i][j] * u[j]
		u[i] = (v[i] - sum) / U[i][i]

	print("discrepancy:", get_vec_norm(f - (L.dot(U)).dot(u)))
	print("LU method:\n", u)

def seidel_next_seq_mem(u_k, L, U, D, f):

	L_plus_D_inv = np.linalg.inv(L + D)
	return -(L_plus_D_inv.dot(U)).dot(u_k) + L_plus_D_inv.dot(f)

def seidel_method(matrix, f):
	
	u_k = np.zeros(n)
	u_k_plus_1 = np.zeros(n)

	L = np.zeros((n, n))
	U = np.zeros((n, n))
	D = np.zeros((n, n))

	for i in range(n):
		for j in range(i):
			L[i][j] = matrix[i][j]
	for i in range(n):
		for j in range(i + 1, n):
			U[i][j] = matrix[i][j]
	for i in range(n):
		D[i][i] = matrix[i][i]

	i = 0
	while get_vec_norm(f - matrix.dot(u_k)) >= eps:
		i += 1
		u_k = seidel_next_seq_mem(u_k, L, U, D, f)

	print("discrepancy:", get_vec_norm(f - matrix.dot(u_k)))
	print(i, ":\n", u_k)

def high_relaxation_method(matrix, f):

	omega = 1.05
	u_k = np.zeros(n)
	u_k_plus_1 = np.zeros(n)
	z_k = np.zeros(n)

	L = np.zeros((n, n))
	U = np.zeros((n, n))
	D = np.zeros((n, n))

	for i in range(n):
		for j in range(i):
			L[i][j] = matrix[i][j]
	for i in range(n):
		for j in range(i + 1, n):
			U[i][j] = matrix[i][j]
	for i in range(n):
		D[i][i] = matrix[i][i]

	i = 0
	while get_vec_norm(f - matrix.dot(u_k)) >= eps:
		i += 1
		z_k = seidel_next_seq_mem(z_k, L, U, D, f) # z k+1
		u_k = u_k * (1 - omega) + omega * z_k

	print("discrepancy:", get_vec_norm(f - matrix.dot(u_k)))
	print(i, ":\n", u_k) 

matrix = create_matrix()
print(matrix)
print("max lambda:", get_max_lambda(matrix))
print("min lambda:", get_min_lambda(matrix))
lambdas, v = np.linalg.eig(matrix)
lambdas.sort()
print("true max lambda:\n", lambdas[99])
print("true min lambda:\n", lambdas[0])
compute_matrix_condition_num(matrix)
f = create_f_vector()
ret = check_LU(matrix)
L, U = LU_factorization(matrix)
solve_LU(L, U, f)
#seidel_method(matrix, f)
high_relaxation_method(matrix, f)