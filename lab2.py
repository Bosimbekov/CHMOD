import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3

def trapezoidal_method(a, b, n):
    """
    Численное интегрирование методом трапеций.
    Параметры:  
    a, b - границы интегрирования
    n - число разбиений     
    Возвращает значение интеграла.  
    """   
    h = (b - a)/n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    integral = (h/2) * (y[0] + 2 * sum(y[1: -1]) + y[-1])

    return integral

# Настоящее значение интеграла для f(x) = x^3 в пределах [1, 4]
def true_integral(a, b):
    return (b**4)/4 - (a**4)/4

# Границы интегрирования
a, b = 1, 4
true_value = true_integral(a, b)

# Число разбиений для исследования
n_values =[ 10, 20, 40 ,80,160]
errors = []
runge_estimations = []

for n in n_values:
    I_n = trapezoidal_method(a, b, n)
    I_2n = trapezoidal_method(a, b, 2 * n)
    error = abs(true_value - I_n)
    errors.append(error)
    # Оценка ошибки по правилу Рунге
    runge_error = abs (I_2n-I_n)/3
    runge_estimations.append(runge_error)
    
    # Вывод значения интеграла для текущего числа разбиений
    print(f"Число разбиений: {n}, значение интеграла: {I_n}")

# Построение графика истинной погрешности
plt.figure(figsize=(10, 6))
plt.loglog(n_values, errors, '-o', label='Истинная погрешность')
plt.loglog(n_values, runge_estimations, '-s', label='Оценка ошибки по правилу Рунге')
plt.xlabel('Число разбиений (n)')
plt.ylabel('Погрешность')
plt.title('Зависимость истинной погрешности и оценки по правилу Рунге от числа разбиений')
plt.legend()
plt.grid(which="both", linestyle ="--", linewidth=0.5)
plt.show()

# Сравнение ошибок для трех значений числа разбиений
print("Число разбиений (n):", n_values[:3])
print("Истинные ошибки:", errors[:3])
print("Оценки ошибок по правилу Рунге:", runge_estimations[:3])                                                                                                                                                                                                                                                 