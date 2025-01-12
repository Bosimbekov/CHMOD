import numpy as np
import matplotlib.pyplot as plt

# Функция для уравнения f(x) = x^4 - 13*x^2 + 36
def f(x):
    return x**4 - 13*x**2 + 36

# Производная функции f(x)
def df(x):
    return 4*x**3 - 26*x

# Метод Ньютона-Рафсона
def newton_raphson(f, df, x0, tol=1e-6, max_iter=1000):
    x = x0
    errors = []
    for i in range(max_iter):
        x_new = x - f(x) / df(x)
        error = abs(x_new - x)
        errors.append(error)
        if error < tol:
            break
        x = x_new
    return x, errors

# Начальное приближение
x0 = 10

# Поиск корня и погрешности
root, errors = newton_raphson(f, df, x0)

# Количество итераций
iterations = len(errors)

# Печатаем результаты
print(f"Найденный корень: {root}")
print(f"Число итераций: {iterations}")
print(f"Истинная погрешность: {errors[-1]}")

# Построение графика зависимости погрешности от числа итераций
plt.plot(range(1, iterations + 1), errors)
plt.yscale('log')  # Логарифмическая шкала для погрешности
plt.xlabel('Число итераций')
plt.ylabel('Погрешность')
plt.title('Зависимость погрешности от числа итераций')
plt.grid(True)
plt.show()