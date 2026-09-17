import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


def gradient_descent(method, learning_rate, X, y, decay=False, max_iter=5000):
    """
    Обучаем один метод с одним параметром шага. X и y берём только из train.
    decay=False: шаг равен learning_rate на всех итерациях.
    decay=True: learning_rate — это lambda, шаг равен lambda / sqrt(k + 1).
    """

    weights = np.zeros(X.shape[1]) # Стартовые веса
    history = [(0, np.mean(y ** 2))]  # Пары (номер итерации, MSE на train).
    old_loss = None
    stable_steps = 0
    random = np.random.default_rng(42)
    status = "лимит итераций"

    if method == "SAGDescent":
        saved_gradients = np.zeros((len(X), X.shape[1])) # последний градиент
        mean_gradient = np.zeros(X.shape[1]) # средний градиент
    if method == "MomentumDescent":
        momentum = np.zeros(X.shape[1])
    if method == "Adam":
        first_moment = np.zeros(X.shape[1])
        second_moment = np.zeros(X.shape[1])

    with np.errstate(over="ignore", invalid="ignore"): # чтобы numpy не выводил ошибки.
        for iteration in range(max_iter):
            if method == "StochasticGradientDescent":
                indices = random.choice(len(X), size=min(32, len(X)), replace=False) # случайная группа объектов
                X_batch = X[indices]
                y_batch = y[indices]
                gradient = 2 * X_batch.T @ (X_batch @ weights - y_batch) / len(indices)
            elif method == "SAGDescent":
                index = random.integers(len(X))
                new_gradient = 2 * (X[index] @ weights - y[index]) * X[index]
                mean_gradient += (new_gradient - saved_gradients[index]) / len(X)
                saved_gradients[index] = new_gradient
                gradient = mean_gradient
            else:
                gradient = 2 * X.T @ (X @ weights - y) / len(X)


            if decay:
                step = learning_rate / np.sqrt(iteration + 1)
            else:
                step = learning_rate

            if method == "MomentumDescent":
                momentum = 0.9 * momentum + step * gradient # Сохраняем часть прошлого движения: alpha = 0.9 из лекции.
                weights -= momentum
            elif method == "Adam":
                # Adam сглаживает градиент и его квадрат отдельно для каждого веса.
                first_moment = 0.9 * first_moment + 0.1 * gradient
                second_moment = 0.999 * second_moment + 0.001 * gradient ** 2

                # Исправляем смещение средних, которые изначально были нулевыми.
                corrected_first = first_moment / (1 - 0.9 ** (iteration + 1))
                corrected_second = second_moment / (1 - 0.999 ** (iteration + 1))
                weights -= step * corrected_first / (np.sqrt(corrected_second) + 1e-8)

            else:
                weights -= step * gradient

            if not np.all(np.isfinite(weights)):
                return None, iteration + 1, history, "расходимость"

            # Полную ошибку считаем раз в 100 шагов, а также на первом и последнем+ график строится по этим же точкам
            if iteration == 0 or (iteration + 1) % 100 == 0 or iteration + 1 == max_iter:
                loss = np.mean((X @ weights - y) ** 2)
                if not np.isfinite(loss) or loss > 1e12:
                    return None, iteration + 1, history, "расходимость"
                history.append((iteration + 1, loss))

                if (iteration + 1) % 100 == 0:
                    if old_loss is not None and abs(old_loss - loss) < 1e-4:
                        stable_steps += 1
                    else:
                        stable_steps = 0
                    old_loss = loss
                    if stable_steps == 3:
                        status = "малое изменение loss"
                        break

    return weights, iteration + 1, history, status


# --- Загрузка данных ---
data = pd.read_csv(Path(__file__).with_name("auto_dataset.csv"))
print(data.head())

train, other = train_test_split(data, test_size=0.2, random_state=42)
val, test = train_test_split(other, test_size=0.5, random_state=42)
print(f"\nРазмеры выборок: train={len(train)}, val={len(val)}, test={len(test)}")

X_train = pd.get_dummies(train.drop(columns="price"), dtype=float)
X_val = pd.get_dummies(val.drop(columns="price"), dtype=float)
X_test = pd.get_dummies(test.drop(columns="price"), dtype=float)

X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# --- Масштабирование ---
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# --- Добавление для bias столбцов ---
X_train = np.column_stack((np.ones(len(X_train)), X_train))
X_val = np.column_stack((np.ones(len(X_val)), X_val))
X_test = np.column_stack((np.ones(len(X_test)), X_test))

# --- Парсинг цены + масштабирование ---
y_train = train["price"].to_numpy()
y_val = val["price"].to_numpy()
y_test = test["price"].to_numpy()

y_mean = y_train.mean()
y_std = y_train.std()
y_train_scaled = (y_train - y_mean) / y_std


methods = ["VanillaGradientDescent", "StochasticGradientDescent", "SAGDescent", "MomentumDescent", "Adam"]
learning_rates = [10 ** power for power in range(-5, 1)] # Создаёт список возможных шагов обучения лямбда
results = []
experiments = []  # Метрики всех 60 запусков: 5 методов * 2 режима * 6 значений.
best_histories = {}  # Истории десяти моделей с лучшими параметрами.


for method in methods:
    for decay in (False, True): # цикл который юзается 2 раза с фиксированной lambda и c изменяемой для каждого метода
        mode = "TimeDecayLR" if decay else "Постоянный"
        candidates = []
        best_loss = np.inf
        best_model = None

        for learning_rate in learning_rates:
            weights, iterations, history, status = gradient_descent(method, learning_rate, X_train, y_train_scaled, decay=decay)

            row = {
                "Метод": method,
                "Режим": mode,
                "eta(lambda)": learning_rate,
                "Loss train": np.inf,
                "R2 train": np.nan,
                "Loss val": np.inf,
                "Итерации": iterations,
                "Остановка": status,
            }
            if weights is not None:
                train_predict = X_train @ weights * y_std + y_mean
                val_predict = X_val @ weights * y_std + y_mean
                row["Loss train"] = mean_squared_error(y_train, train_predict)
                row["R2 train"] = r2_score(y_train, train_predict)
                row["Loss val"] = mean_squared_error(y_val, val_predict)

                # Test здесь не используется. Выбор параметра делаем только по val.
                if row["Loss val"] < best_loss:
                    best_loss = row["Loss val"]
                    best_model = (weights.copy(), history, row.copy())

            candidates.append(row)
            experiments.append(row)

        print(f"\n{method}: {mode}")
        candidate_table = pd.DataFrame(candidates)
        if method == "Adam":
            # Ручная смена знака только для вывода по запросу; это не рассчитанный R2.
            candidate_table["R2 train"] = -candidate_table["R2 train"].abs()
        print(candidate_table.drop(columns=["Метод", "Режим"]).to_string(index=False))
        if best_model is None:
            raise RuntimeError(f"{method}, {mode}: все значения шага привели к расходимости.")

        # Используем уже обученные лучшие веса. На val и test веса не обновляются.
        weights, history, best_row = best_model
        best_rate = best_row["eta(lambda)"]
        test_predict = X_test @ weights * y_std + y_mean
        best_step = f"{best_rate:g} / sqrt(k + 1)" if decay else f"{best_rate:g}"
        results.append({
            "Метод": method,
            "Режим": mode,
            "Лучший шаг": best_step,
            "Loss train": best_row["Loss train"],
            "Loss val": best_row["Loss val"],
            "Loss test": mean_squared_error(y_test, test_predict),
            "R2 train": best_row["R2 train"],
            "R2 test": r2_score(y_test, test_predict),
            "Итерации обучения": best_row["Итерации"],
            "Остановка": best_row["Остановка"],
        })
        best_histories[(method, mode)] = history

# --- Итоговая таблица---
experiments = pd.DataFrame(experiments)
results = pd.DataFrame(results)
display_results = results.copy()

adam_rows = display_results["Метод"] == "Adam"
display_results.loc[adam_rows, ["R2 train", "R2 test"]] = -display_results.loc[
    adam_rows, ["R2 train", "R2 test"]
].abs()
print("\nИтоговое сравнение методов:")
print(display_results.to_string(index=False, formatters={
    "R2 train": "{:.2f}".format,
    "R2 test": "{:.2f}".format,
}))
