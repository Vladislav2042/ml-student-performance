import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

df = pd.read_pickle("processed_data.pkl")
X = df.drop(columns=['GPA_class'])
y = df['GPA_class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n=== {name} ===")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    return macro_f1


print("\n=== Baseline Models ===")
evaluate_model(LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42), "Logistic Regression")
evaluate_model(DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=42), "Decision Tree")
evaluate_model(RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced_subsample',
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
), "Random Forest")