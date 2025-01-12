import numpy as np
import matplotlib.pyplot as plt

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - np.sum(L[i, :j] ** 2))
            else:
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

    return L

def cholesky_solve(A, b):
    L = cholesky_decomposition(A)
    
    # Решение L * y = b (прямой ход)
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = (b[i] - np.sum(L[i, :i] * y[:i])) / L[i, i]

    # Решение L^T * x = y (обратный ход)
    x = np.zeros_like(b)
    for i in range(len(b) - 1, -1, -1):
        x[i] = (y[i] - np.sum(L[i + 1:, i] * x[i + 1:])) / L[i, i]

    return x

def solve_slae_iteratively(A, b, tolerance=1e-6):
    n = len(b)
    x = np.zeros(n)  # Начальное приближение
    x_prev = np.copy(x)
    iterations = []
    solutions = []

    while True:
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_prev[i + 1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        iterations.append(len(iterations) + 1)
        solutions.append(np.copy(x))

        if np.linalg.norm(x - x_prev, ord=np.inf) < tolerance:
            break

        x_prev = np.copy(x)

    return np.array(solutions), iterations

# Матрица A и вектор b
A = np.array([
    [10, 1, 1],
    [2, 10 + 1, 1],  # A = 1
    [2, 2, 10]
], dtype=float)

b = np.array([12, 13, 14], dtype=float)

# Решение методом Холецкого
x_cholesky = cholesky_solve(A, b)
print("Решение методом Холецкого:", x_cholesky)

# Итерационное решение (например, методом Якоби)
solutions, iterations = solve_slae_iteratively(A, b)

# Построение графика
solutions = np.array(solutions)
plt.figure(figsize=(10, 6))
for i in range(solutions.shape[1]):
    plt.plot(iterations, solutions[:, i], label=f"x{i + 1}")

plt.xlabel("Итерации")
plt.ylabel("Значение корней")
plt.title("Зависимость значений корней от номера итерации")
plt.legend()
plt.grid()
plt.show()
