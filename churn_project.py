import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Ensure directories exist
os.makedirs('models', exist_ok=True)

# 1. Load Dataset from 'data' folder
print("Loading dataset...")
df = pd.read_csv('data/Customer-Churn-Records.csv')

# Drop unnecessary identifier columns
drop_cols = [col for col in ['RowNumber', 'CustomerId', 'Surname'] if col in df.columns]
df = df.drop(columns=drop_cols)

# 2. Preprocessing & Encoding Categorical Columns
print("Preprocessing data...")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Define Features (X) and Target (y)
target_column = 'Exited' if 'Exited' in df.columns else 'Churn'
X = df.drop(columns=[target_column])
y = df[target_column]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Define Models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

model_performances = {}
trained_models = {}

print("\n--- Training Models ---")
for name, model in models.items():
    if name in ['K-Nearest Neighbors', 'SVM']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    model_performances[name] = acc
    trained_models[name] = model
    print(f"{name} -> Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")

# 6. Identify & Save Best Model
best_model_name = max(model_performances, key=model_performances.get)
print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {model_performances[best_model_name]:.4f})")

# Save artifacts inside 'models/' folder
joblib.dump(trained_models[best_model_name], 'models/best_churn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(list(X.columns), 'models/feature_columns.pkl')
print("Model, scaler, and features saved successfully inside 'models/' folder!")