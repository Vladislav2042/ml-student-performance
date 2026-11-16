import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"/global_university_students_performance_habits_10000.csv")

print("Размер данных:", df.shape)
print("\nНазвания всех колонок:")
print(df.columns.tolist())
print("\nПервые 5 строк:")
print("\nИнформация о данных:")
print(df.info())
print("\nСтатистика по GPA:")
print(df['GPA'].describe())
# Удаляем final_exam_score
df = df.drop(columns=['final_exam_score'])
df = df.drop(columns=['student_id'])
print("Колонки после удаления:")
print(df.columns.tolist())
print("\nНовый размер данных:", df.shape)



# Создаём новую колонку с классами
df['GPA_class'] = pd.cut(
    df['GPA'],
    bins=[-np.inf, 3.4, 3.8, np.inf],
    labels=['Low', 'Medium', 'High']
)

print("Распределение классов:")
print(df['GPA_class'].value_counts().sort_index())

print("\nПроцентное распределение:")
print(df['GPA_class'].value_counts(normalize=True).sort_index() * 100)
