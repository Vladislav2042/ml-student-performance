import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv(r"/global_university_students_performance_habits_10000.csv")
print("Исходный размер:", df.shape)

# 1. Создаём целевую переменную GPA_class (если её ещё нет)
if 'GPA_class' not in df.columns:
    df['GPA_class'] = pd.cut(
        df['GPA'],
        bins=[-np.inf, 3.4, 3.8, np.inf],
        labels=['Low', 'Medium', 'High']
    )

# 2. Удаляем ненужные колонки
cols_to_drop = ['student_id', 'GPA', 'final_exam_score']
df = df.drop(columns=cols_to_drop, errors='ignore')

print("После удаления ненужных колонок:", df.shape)

# 3. Обработка пропусков
df['favorite_AI_tool'] = df['favorite_AI_tool'].fillna('None')

# 4. Кодируем все категориальные признаки
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\nКатегориальные колонки для кодирования:")
print(categorical_cols)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 5. Разделяем на X и y
X = df.drop(columns=['GPA_class'])
y = df['GPA_class']

print("\n=== Итоговые размеры ===")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nРаспределение классов:")
print(y.value_counts().sort_index())

print("\nТипы данных в X:")
print(X.dtypes.value_counts())
df.to_pickle("processed_data.pkl")

print("\n=== Данные успешно сохранены в processed_data.pkl ===")
print("Теперь в следующих запусках можно загружать уже готовые данные.")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ============================================
# 1. Разделение данных
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # важно при дисбалансе
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
print("Распределение в train:\n", y_train.value_counts(normalize=True))


# ============================================
# 2. Функция для обучения и оценки модели
# ============================================
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Основная метрика
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n{'=' * 50}")
    print(f"Модель: {model_name}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    return macro_f1


# ============================================
# 3. Обучение моделей с class_weight
# ============================================

# === Logistic Regression ===
log_reg = LogisticRegression(
    class_weight='balanced',
    max_iter=2000,
    random_state=42
)
evaluate_model(log_reg, "Logistic Regression")

# === Decision Tree ===
dt = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=10,
    random_state=42
)
evaluate_model(dt, "Decision Tree")

# === Random Forest (рекомендуется) ===
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced_subsample',  # лучший вариант для Random Forest
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
evaluate_model(rf, "Random Forest")
#==========#
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
import numpy as np

# ============================================
# 1. Определяем метрику (macro F1)
# ============================================
f1_macro_scorer = make_scorer(f1_score, average='macro')

# ============================================
# 2. Определяем пространство поиска гиперпараметров
# ============================================
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample']
}

# ============================================
# 3. Настраиваем кросс-валидацию
# ============================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================
# 4. Создаём RandomizedSearchCV
# ============================================
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=30,                    # сколько комбинаций проверить (можно поставить 50)
    scoring=f1_macro_scorer,      # оптимизируем macro F1
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# ============================================
# 5. Запускаем поиск
# ============================================
print("Начинаем подбор гиперпараметров...")
random_search.fit(X_train, y_train)

print("\nЛучшие гиперпараметры:")
print(random_search.best_params_)

print(f"\nЛучший Macro F1 на кросс-валидации: {random_search.best_score_:.4f}")

# ============================================
# 6. Проверяем лучшую модель на тестовой выборке
# ============================================
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)

test_macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\nMacro F1 на тестовой выборке: {test_macro_f1:.4f}")

print("\nClassification Report (лучшая модель):")
print(classification_report(y_test, y_pred, digits=4))