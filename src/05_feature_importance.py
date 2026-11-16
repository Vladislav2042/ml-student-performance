import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

saved = joblib.load("best_xgb_model.pkl")
model = saved['model']

df = pd.read_pickle("processed_data.pkl")
X = df.drop(columns=['GPA_class'])

importances = model.feature_importances_
feature_names = X.columns

feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\nTop 15 Most Important Features:\n")
print(feat_imp.head(15).to_string(index=False))

plt.figure(figsize=(12, 10))
sns.barplot(data=feat_imp.head(15), x='importance', y='feature', palette='viridis')
plt.title('Top 15 Feature Importance (XGBoost)', fontsize=16)
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()