import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Исходные данные
x_data = np.array([289, 334, 300, 343, 356, 289, 341, 327, 357, 352, 381])
y_data = np.array([6.9, 8.7, 6.4, 8.4, 6.1, 9.4, 11.0, 6.4, 9.3, 8.2, 8.6])

# Определение функций для моделей
def linear_func(x, a, b):
    return a * x + b

def exponential_func(x, a, b):
    return a * np.exp(b * x)

def exp_func(x, a, b):
    return a * np.exp(b / x)

def sqrt_func(x, a, b):
    return a * np.sqrt(x) + b

def polynomial_func(x, a, b, c):
    return a * x**2 + b * x + c

def hyperbolic_func(x, a, b):
    return a / x + b

def power_func(x, a, b):
    return a * x**b

# Список моделей
models = [
    {'name': 'Линейная', 'func': linear_func, 'p0': [1, 1], 'params': None, 'y_pred': None},
    {'name': 'Показательная', 'func': exponential_func, 'p0': [1, 0.001], 'params': None, 'y_pred': None},
    {'name': 'Экспоненциальная', 'func': exp_func, 'p0': [1, 1], 'params': None, 'y_pred': None},
    {'name': 'Корень', 'func': sqrt_func, 'p0': [1, 1], 'params': None, 'y_pred': None},
    {'name': 'Полиномиальная', 'func': polynomial_func, 'p0': [1, 1, 1], 'params': None, 'y_pred': None},
    {'name': 'Гиперболическая', 'func': hyperbolic_func, 'p0': [1, 1], 'params': None, 'y_pred': None},
    {'name': 'Степенная', 'func': power_func, 'p0': [1, 1], 'params': None, 'y_pred': None}
]

# Подгонка моделей
for model in models:
    try:
        params, _ = curve_fit(model['func'], x_data, y_data, p0=model['p0'], maxfev=5000)
        model['params'] = params
        model['y_pred'] = model['func'](x_data, *params)
    except Exception as e:
        print(f"Ошибка при подгонке модели {model['name']}: {e}")
        model['y_pred'] = np.zeros_like(y_data)

# Функции для расчета показателей
def calculate_correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

def calculate_determination(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_elasticity(x, y_pred, func, params):
    # Средний коэффициент эластичности
    x_mean = np.mean(x)
    y_pred_mean = np.mean(y_pred)
    
    # Численное вычисление производной
    h = 0.001
    y_plus = func(x_mean + h, *params)
    y_minus = func(x_mean - h, *params)
    derivative = (y_plus - y_minus) / (2 * h)
    
    elasticity = derivative * (x_mean / y_pred_mean)
    return abs(elasticity)

def calculate_fisher(y_true, y_pred, k):
    n = len(y_true)
    ssr = np.sum((y_pred - np.mean(y_true))**2)
    sse = np.sum((y_true - y_pred)**2)
    
    if sse == 0:
        return float('inf')
    
    f_stat = (ssr / k) / (sse / (n - k - 1))
    return f_stat

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Расчет показателей для всех моделей
print("РЕЗУЛЬТАТЫ РЕГРЕССИОННОГО АНАЛИЗА")
print("=" * 100)

for i, model in enumerate(models):
    if model['y_pred'] is not None:
        # Основные показатели
        correlation = calculate_correlation(y_data, model['y_pred'])
        determination = calculate_determination(y_data, model['y_pred'])
        mape = calculate_mape(y_data, model['y_pred'])
        
        # Количество параметров для критерия Фишера
        k = len(model['params'])
        f_stat = calculate_fisher(y_data, model['y_pred'], k)
        
        # Коэффициент эластичности
        elasticity = calculate_elasticity(x_data, model['y_pred'], model['func'], model['params'])
        
        print(f"\n{model['name']} модель:")
        print(f"Параметры уравнения: {model['params']}")
        print(f"Коэффициент корреляции: {correlation:.4f}")
        print(f"Коэффициент детерминации R²: {determination:.4f}")
        print(f"Коэффициент эластичности: {elasticity:.4f}")
        print(f"F-критерий Фишера: {f_stat:.4f}")
        print(f"Средняя ошибка аппроксимации (MAPE): {mape:.2f}%")

# Построение графика
plt.figure(figsize=(12, 8))
plt.scatter(x_data, y_data, color='blue', s=80, label='Исходные данные', zorder=5)

# Цвета для линий тренда
colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

# Создание равномерно распределенных точек для гладких кривых
x_smooth = np.linspace(min(x_data), max(x_data), 300)

# Построение линий тренда
for i, model in enumerate(models):
    if model['params'] is not None:
        try:
            y_smooth = model['func'](x_smooth, *model['params'])
            plt.plot(x_smooth, y_smooth, color=colors[i], linewidth=2, 
                    label=f"{model['name']}", linestyle='--' if i > 0 else '-')
        except:
            continue

plt.xlabel('Среднемесячная начисленная заработная плата, тыс. руб.', fontsize=12)
plt.ylabel('Доля денежных доходов, направленных на прирост сбережений, %', fontsize=12)
plt.title('Поле корреляции и модели регрессии', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Добавление названий точек
regions = ['Брянская', 'Владимирская', 'Ивановская', 'Калужская', 'Костромская', 
           'Орловская', 'Рязанская', 'Смоленская', 'Тверская', 'Тульская', 'Ярославская']

for i, (x, y, region) in enumerate(zip(x_data, y_data, regions)):
    plt.annotate(region, (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=8, alpha=0.7)

plt.show()

# Сравнительная таблица результатов
print("\n" + "=" * 100)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 100)
print(f"{'Модель':<15} {'R²':<8} {'MAPE,%':<10} {'F-критерий':<12} {'Эластичность':<12}")
print("-" * 60)

for model in models:
    if model['y_pred'] is not None:
        determination = calculate_determination(y_data, model['y_pred'])
        mape = calculate_mape(y_data, model['y_pred'])
        k = len(model['params'])
        f_stat = calculate_fisher(y_data, model['y_pred'], k)
        elasticity = calculate_elasticity(x_data, model['y_pred'], model['func'], model['params'])
        
        print(f"{model['name']:<15} {determination:.4f}  {mape:.2f}      {f_stat:.2f}        {elasticity:.4f}")

# Выбор лучшей модели
best_model = None
best_score = float('inf')

for model in models:
    if model['y_pred'] is not None:
        mape = calculate_mape(y_data, model['y_pred'])
        if mape < best_score:
            best_score = mape
            best_model = model

print(f"\nЛучшая модель: {best_model['name']}")
print(f"С наименьшей ошибкой аппроксимации: {best_score:.2f}%")