import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("global_university_students_performance_habits_10000.csv")
print(f"Original size: {df.shape}")

if 'GPA_class' not in df.columns:
    df['GPA_class'] = pd.cut(
        df['GPA'],
        bins=[-np.inf, 2.5, 3.5, np.inf],
        labels=['Low', 'Medium', 'High']
    )

cols_to_drop = ['student_id', 'GPA', 'final_exam_score']
df = df.drop(columns=cols_to_drop, errors='ignore')
print(f"After removing unnecessary columns: {df.shape}")

df['favorite_AI_tool'] = df['favorite_AI_tool'].fillna('None')

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns for encoding: {categorical_cols}")

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=['GPA_class'])
y = df['GPA_class']

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nClass distribution:\n{y.value_counts().sort_index()}")

df.to_pickle("processed_data.pkl")
print("\nProcessed data successfully saved to processed_data.pkl")