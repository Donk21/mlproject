# Импорт библиотек
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance


# Загрузка и подготовка данных
data = pd.read_csv('customer_basket_data.csv')
X = data.drop('target', axis=1)
y = data['target']

categorical_features = ['category', 'day_of_week', 'time_of_day']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

X = data.drop(columns=['customer_id', 'product_id', 'target'])
y = data['target']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Случайный лес
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
rf_f1 = f1_score(y_test, rf_preds)

# Градиентный бустинг
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
xgb_f1 = f1_score(y_test, xgb_preds)


# Нейронные сети
nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=200, random_state=42)
nn_model.fit(X_train, y_train)
nn_preds = nn_model.predict(X_test)
nn_auc = roc_auc_score(y_test, nn_model.predict_proba(X_test)[:, 1])
nn_f1 = f1_score(y_test, nn_preds)

# Сравнение моделей
print("Random Forest: AUC = {:.2f}, F1 = {:.2f}".format(rf_auc, rf_f1))
print("XGBoost: AUC = {:.2f}, F1 = {:.2f}".format(xgb_auc, xgb_f1))
print("Neural Network: AUC = {:.2f}, F1 = {:.2f}".format(nn_auc, nn_f1))

# Проверка размеров выборок
print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

# Проверка распределения целевой переменной
print("Распределение целевой переменной в обучающей выборке:")
print(y_train.value_counts(normalize=True))
print("Распределение целевой переменной в тестовой выборке:")
print(y_test.value_counts(normalize=True))

# Проверка распределения классов
print("Распределение классов:")
print(y.value_counts())
min_class_size = y.value_counts().min()
print(f"Минимальное количество элементов в классе: {min_class_size}")

# Установить количество разбиений в StratifiedKFold, не превышающее минимальный размер класса
cv_splits = min(3, min_class_size)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Проверяем распределение классов в обучающей выборке
print("Распределение классов в обучающей выборке:")
print(y_train.value_counts())

# Пример первичного обучения для случайного леса
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("Начальная точность случайного леса:", rf_model.score(X_test, y_test))

# Настройка гиперпараметров для случайного леса
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Использование StratifiedKFold
stratified_kf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

# Создание и запуск GridSearchCV
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=stratified_kf,
    verbose=1,  # Вывод процесса обучения
    n_jobs=-1  # Использование всех доступных ядер процессора
)
grid_search.fit(X_train, y_train)

# Результаты GridSearchCV
print("Лучшие параметры случайного леса:", grid_search.best_params_)
print("Лучший ROC-AUC на обучении:", grid_search.best_score_)

# Обучение модели с оптимальными параметрами
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train, y_train)

# Оценка производительности на тестовой выборке
y_test_preds = best_rf_model.predict(X_test)
y_test_proba = best_rf_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_proba)
print("ROC-AUC на тестовой выборке:", test_auc)

# Список моделей для оценки
models = {
    "Random Forest": best_rf_model,
    "XGBoost": xgb_model,
    "Neural Network": nn_model
}

# Словарь для хранения метрик
metrics = {}

# Расчет метрик для каждой модели
for model_name, model in models.items():
    print(f"\n--- Оценка модели: {model_name} ---")

    # Предсказания
    if model_name == "Neural Network":
        # Нейронная сеть возвращает вероятности, преобразуем в метки
        y_test_proba = model.predict(X_test).flatten()
        y_test_preds = (y_test_proba > 0.5).astype(int)
    else:
        y_test_preds = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

    # Вычисление метрик
    accuracy = accuracy_score(y_test, y_test_preds)
    f1 = f1_score(y_test, y_test_preds)
    roc_auc = roc_auc_score(y_test, y_test_proba)

    # Сохранение метрик
    metrics[model_name] = {
        "Accuracy": accuracy,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }

    # Вывод метрик
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_test_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # ROC-кривая
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"ROC Curve for {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()

# Сравнительная таблица метрик
metrics_df = pd.DataFrame(metrics).T
print("\n--- Сравнительная таблица метрик ---")
print(metrics_df)

# Сохранение результатов в CSV
metrics_df.to_csv("model_metrics.csv", index=True)

# Важность признаков для градиентного бустинга
xgb_importances = pd.DataFrame({
    "Признак": X_train.columns,
    "Важность": xgb_model.feature_importances_
}).sort_values(by="Важность", ascending=False)

# Визуализация важности признаков
plt.figure(figsize=(10, 6))
plt.barh(xgb_importances["Признак"], xgb_importances["Важность"])
plt.title("Важность признаков (XGBoost)")
plt.xlabel("Важность")
plt.ylabel("Признак")
plt.gca().invert_yaxis()
plt.show()

# SHAP для интерпретации
explainer = shap.Explainer(xgb_model, X_test)
shap_values = explainer(X_test)

# Визуализация SHAP значений
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)