import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib

saved = joblib.load("best_xgb_model.pkl")
model = saved['model']
le_target = saved['label_encoder']

df = pd.read_pickle("processed_data.pkl")
X = df.drop(columns=['GPA_class'])
y = df['GPA_class']

y_encoded = le_target.transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
class_names = le_target.classes_

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - XGBoost')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

print("\n" + "="*60)
print("MEDIUM CLASS ERROR ANALYSIS")
print("="*60)

results = pd.DataFrame({
    'True': le_target.inverse_transform(y_test),
    'Predicted': le_target.inverse_transform(y_pred)
})

medium_errors = results[(results['True'] == 'Medium') & (results['Predicted'] != 'Medium')]

print(f"\nTotal Medium students in test set: {len(results[results['True'] == 'Medium'])}")
print(f"Incorrectly predicted: {len(medium_errors)}")

print("\nMedium students were misclassified as:")
print(medium_errors['Predicted'].value_counts())

false_medium = results[(results['Predicted'] == 'Medium') & (results['True'] != 'Medium')]
print(f"\nStudents incorrectly classified as Medium: {len(false_medium)}")
print(false_medium['True'].value_counts())