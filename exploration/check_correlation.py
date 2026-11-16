import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Твой датафрейм ===
df = pd.read_csv(r"/global_university_students_performance_habits_10000.csv")

# 1. Корреляция Пирсона (линейная связь)
corr_pearson = df[['GPA', 'final_exam_score', 'assignment_score']].corr(method='pearson')
print("Корреляция Пирсона:")
print(corr_pearson)

print("\n" + "="*50 + "\n")

# 2. Корреляция Спирмена (монотонная связь, более устойчива к выбросам)
corr_spearman = df[['GPA', 'final_exam_score', 'assignment_score']].corr(method='spearman')
print("Корреляция Спирмена:")
print(corr_spearman)

# 3. Визуализация
plt.figure(figsize=(8, 6))
sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', center=0, fmt=".3f")
plt.title("Корреляция Пирсона между GPA и оценками")
plt.show()