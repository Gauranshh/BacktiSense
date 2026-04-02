import os
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load & preprocess ──────────────────────────────────────────
dataset_path = os.path.join(BASE_DIR, "Dataset.xlsx")
df = pd.read_excel(dataset_path)
df = pd.get_dummies(df, columns=['Location'])

antibiotics = ['IMIPENEM', 'CEFTAZIDIME', 'GENTAMICIN', 'AUGMENTIN', 'CIPROFLOXACIN']

def convert(value):
    return 0 if value >= 20 else 1   # 0 = Susceptible, 1 = Resistant

for col in antibiotics:
    df[col] = df[col].apply(convert)

X = df.drop(columns=antibiotics)
y = df[antibiotics]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train model ────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────
y_pred = model.predict(X_test)
print("=== Model Performance ===")
for i, ab in enumerate(antibiotics):
    acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{ab}: {round(acc * 100, 2)}% accuracy")

# ── Save model ─────────────────────────────────────────────────
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)
print(f"\nModel saved to {model_path}")

# ── Export predictions per location for frontend ───────────────
locations = [col.replace("Location_", "") for col in X.columns if "Location_" in col]
output = {}

for loc in locations:
    input_dict = {col: 0 for col in X.columns}
    for col in X.columns:
        if loc in col:
            input_dict[col] = 1
    sample = pd.DataFrame([input_dict])
    
    pred = model.predict(sample)[0]
    probs = model.predict_proba(sample)
    
    output[loc] = {}
    for i, ab in enumerate(antibiotics):
        susceptible_prob = round(probs[i][0][0] * 100, 2)
        output[loc][ab] = [int(pred[i]), susceptible_prob]

predictions_path = os.path.join(model_dir, "predictions.json")
with open(predictions_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Predictions exported to {predictions_path}")

# ── Export feature importances ─────────────────────────────────
importances = model.estimators_[0].feature_importances_
feat_imp = sorted(zip(X.columns.tolist(), importances), key=lambda x: -x[1])
feat_imp_top = feat_imp[:8]

feat_output = [{"name": name.replace("Location_",""), "value": round(float(val)*100, 1)} 
               for name, val in feat_imp_top]

feature_importances_path = os.path.join(model_dir, "feature_importance.json")
with open(feature_importances_path, "w") as f:
    json.dump(feat_output, f, indent=2)
print(f"Feature importance exported to {feature_importances_path}")

base_model = RandomForestClassifier(n_estimators=200, random_state=42)
model = MultiOutputClassifier(base_model)
model.fit(X_train, y_train)

accuracy_dict = {}

for i, ab in enumerate(antibiotics):
    acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    accuracy_dict[ab] = round(acc * 100, 2)

accuracy_path = os.path.join(model_dir, "accuracy.json")

with open(accuracy_path, "w") as f:
    json.dump(accuracy_dict, f, indent=2)

print(f"Accuracy exported to {accuracy_path}")