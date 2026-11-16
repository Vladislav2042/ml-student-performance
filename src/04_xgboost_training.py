import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
import joblib

df = pd.read_pickle("processed_data.pkl")
X = df.drop(columns=['GPA_class'])
y = df['GPA_class']

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

best_score = 0
best_model = None
best_lr = None

for lr in [0.03, 0.05, 0.07]:
    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=lr,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_val_pred = model.predict(X_val)
    score = f1_score(y_val, y_val_pred, average='macro')
    trees_used = model.best_iteration if hasattr(model, 'best_iteration') else model.best_ntree_limit

    print(f"lr={lr}, trees used={trees_used}, Validation Macro F1={score:.4f}")

    if score > best_score:
        best_score = score
        best_model = model
        best_lr = lr

print(f"\nBest learning rate: {best_lr}")
print(f"Best Validation Macro F1: {best_score:.4f}")

y_test_pred = best_model.predict(X_test)
test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')

print(f"\nTest Macro F1: {test_macro_f1:.4f}")

y_test_labels = le_target.inverse_transform(y_test)
y_pred_labels = le_target.inverse_transform(y_test_pred)

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels, digits=4))

joblib.dump(
    {
        'model': best_model,
        'label_encoder': le_target,
        'data_file': 'processed_data.pkl',
        'best_learning_rate': best_lr,
        'validation_macro_f1': best_score,
        'test_macro_f1': test_macro_f1
    },
    "best_xgb_model.pkl"
)

print("\nBest XGBoost model saved to best_xgb_model.pkl")
