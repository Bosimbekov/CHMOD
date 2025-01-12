import numpy as np
import matplotlib.pyplot as plt

def analytical_solution(t, T0, m):
    """Аналитическое решение dT/dt = -mT"""
    return T0 * np.exp(-m * t)

def f(t, T, m):
    """Правая часть ОДУ dT/dt = -mT"""
    return -m * T

def adams_pc2(f, t0, y0, h, n, m):
    """Многошаговый метод Адамса 2-го порядка (Прогноз-Коррекция)."""
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)

    # Начальные условия
    t[0] = t0
    y[0] = y0

    # Первый шаг методом Эйлера
    t[1] = t[0] + h
    y[1] = y[0] + h * f(t[0], y[0], m)

    # Основной цикл
    for i in range(1, n):
        # Прогноз (экстраполяция)
        yp = y[i] + h * f(t[i], y[i], m)

        # Коррекция (интерполяция)
        t[i + 1] = t[i] + h
        y[i + 1] = y[i] + h * 0.5 * (f(t[i], y[i], m) + f(t[i + 1], yp, m))

    return t, y

def modified_euler(f, t0, y0, h, n, m):
    """Модифицированная схема Эйлера."""
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)

    # Начальные условия
    t[0] = t0
    y[0] = y0

    # Основной цикл
    for i in range(n):
        t[i + 1] = t[i] + h
        f1 = f(t[i], y[i], m)
        y_predict = y[i] + h * f1
        y[i + 1] = y[i] + h * 0.5 * (f1 + f(t[i + 1], y_predict, m))

    return t, y

# Параметры задачи
m = 0.0051
T0 = 10000
steps = [100, 50, 25, 10, 5, 2, 1]
t_values = [100, 200, 300, 400, 500]

# Таблицы результатов
results_euler = []
results_adams = []
errors_euler = []
errors_adams = []

for h in steps:
    n = int(max(t_values) / h)

    # Решения методами
    t_euler, y_euler = modified_euler(f, 0, T0, h, n, m)
    t_adams, y_adams = adams_pc2(f, 0, T0, h, n, m)

    # Вычисление значений в нужные моменты времени
    euler_values = [np.interp(t, t_euler, y_euler) for t in t_values]
    adams_values = [np.interp(t, t_adams, y_adams) for t in t_values]

    # Аналитическое решение
    analytical_values = [analytical_solution(t, T0, m) for t in t_values]

    # Погрешности
    euler_errors = [abs(e - a) for e, a in zip(euler_values, analytical_values)]
    adams_errors = [abs(a - a_exact) for a, a_exact in zip(adams_values, analytical_values)]

    # Запись результатов
    results_euler.append(euler_values)
    results_adams.append(adams_values)
    errors_euler.append(euler_errors)
    errors_adams.append(adams_errors)

# Построение графиков
plt.figure(figsize=(10, 6))

# График аналитического решения
t_analytic = np.linspace(0, max(t_values), 1000)
y_analytic = analytical_solution(t_analytic, T0, m)
plt.plot(t_analytic, y_analytic, label="Аналитическое решение", linewidth=2)

# Графики численных методов
for h, euler_values, adams_values in zip(steps, results_euler, results_adams):
    plt.plot(t_values, euler_values, 'o-', label=f'МЭ h={h}')
    plt.plot(t_values, adams_values, 's-', label=f'ПК2 h={h}')

plt.xlabel('Время (с)')
plt.ylabel('Температура (K)')
plt.title('Решение ОДУ методами МЭ и ПК2')
plt.legend()
plt.grid()
plt.show()

# Печать таблицы результатов
print("Результаты (Модифицированная схема Эйлера):")
for h, values in zip(steps, results_euler):
    print(f"h={h}: {values}")

print("\nРезультаты (Метод Адамса ПК2):")
for h, values in zip(steps, results_adams):
    print(f"h={h}: {values}")

print("\nПогрешности (Модифицированная схема Эйлера):")
for h, errors in zip(steps, errors_euler):
    print(f"h={h}: {errors}")

print("\nПогрешности (Метод Адамса ПК2):")
for h, errors in zip(steps, errors_adams):
    print(f"h={h}: {errors}")