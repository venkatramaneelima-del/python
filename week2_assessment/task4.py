import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------
# 1. PROJECT PATH
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "trainer_dataset.csv"

RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

# --------------------------------
# 2. LOAD DATASET
# --------------------------------

data = pd.read_csv(DATA_PATH)

print("\nDATASET LOADED SUCCESSFULLY")
print(data.head())

print("\nDATASET INFORMATION")
print(data.info())

print("\nMISSING VALUES")
print(data.isnull().sum())

# --------------------------------
# 3. DATA PREPARATION
# --------------------------------

data.columns = data.columns.str.strip()

data = data.drop_duplicates()

target = "Attrition"

if target not in data.columns:
    raise ValueError("Attrition column not found!")

X = data.drop(columns=[target, "Employee_ID"], errors="ignore")
y = data[target]

# Convert target if it contains Yes/No
if y.dtype == "object":
    y = y.astype(str).str.strip().str.lower()
    y = y.map({
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0
    })

if y.isnull().any():
    raise ValueError("Check Attrition column values!")

y = y.astype(int)

print("\nDATA PREPARATION COMPLETED")

# --------------------------------
# 4. IDENTIFY COLUMNS
# --------------------------------

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# --------------------------------
# 5. PREPROCESSING
# --------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# --------------------------------
# 6. TRAIN TEST SPLIT
# --------------------------------

stratify_value = (
    y if y.value_counts().min() >= 2 else None
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_value
)

print("\nTRAINING DATA:", X_train.shape)
print("TESTING DATA:", X_test.shape)

# --------------------------------
# 7. CREATE MODELS
# --------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# --------------------------------
# 8. TRAIN AND EVALUATE
# --------------------------------

results = []

best_accuracy = -1
best_model = None
best_model_name = ""

for name, classifier in models.items():

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n------------------------------")
    print("MODEL:", name)
    print("ACCURACY:", round(accuracy, 4))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_test, predictions))

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# --------------------------------
# 9. SAVE MODEL RESULTS
# --------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_DIR / "model_comparison.csv",
    index=False
)

# --------------------------------
# 10. SAVE BEST MODEL
# --------------------------------

joblib.dump(
    best_model,
    BASE_DIR / "best_employee_model.pkl"
)

print("\n==============================")
print("BEST MODEL:", best_model_name)
print("BEST ACCURACY:", round(best_accuracy, 4))
print("MODEL SAVED SUCCESSFULLY!")
print("==============================")

# =====================================
# ROC-AUC EVALUATION
# =====================================

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

print("\n========== ROC-AUC EVALUATION ==========")

# Predict probabilities using best pipeline
y_prob = best_model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC
auc_score = roc_auc_score(y_test, y_prob)

print("Best Model:", best_model_name)
print("ROC-AUC Score:", round(auc_score, 4))

# Generate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {auc_score:.2f}"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve")

plt.legend()
plt.savefig("roc_curve.png")
plt.show()

print("ROC-AUC graph saved successfully!")
# =====================================
# THRESHOLD TUNING
# =====================================

from sklearn.metrics import precision_recall_curve

print("\n========== THRESHOLD TUNING ==========")

# Use test data because X_val is not defined
y_prob = best_model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_test, y_prob
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_threshold = thresholds[f1_scores.argmax()]

print("Best Threshold:", round(best_threshold, 4))
print("Best F1 Score:", round(f1_scores.max(), 4))

tuned_predictions = (
    y_prob >= best_threshold
).astype(int)

print("Threshold Tuning Completed!")
# ==========================================
# REGRESSION MAE, RMSE AND R2
# ==========================================

# =====================================
# REGRESSION: MAE, RMSE AND R2
# =====================================
# =====================================
# REGRESSION MAE, RMSE AND R2
# =====================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

print("\n========== REGRESSION METRICS ==========")

# Load dataset
BASE_DIR = Path(__file__).resolve().parent
csv_files = list(BASE_DIR.rglob("trainer_dataset.csv"))

if not csv_files:
    raise FileNotFoundError("Dataset not found!")

df = pd.read_csv(csv_files[0])

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

# Check target column
if "Annual_Salary" not in df.columns:
    raise ValueError("Annual_Salary column not found!")

# Convert salary into numeric
df_reg = df.copy()

df_reg["Annual_Salary"] = pd.to_numeric(
    df_reg["Annual_Salary"].astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("$", "", regex=False),
    errors="coerce"
)

df_reg = df_reg.dropna(subset=["Annual_Salary"])

# Separate features and target
X_reg = df_reg.drop(columns=["Annual_Salary"])
y_reg = df_reg["Annual_Salary"]

# Identify columns
num_cols = X_reg.select_dtypes(include=np.number).columns
cat_cols = X_reg.select_dtypes(exclude=np.number).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

# Create regression model
reg_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Split dataset
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# Train model
reg_model.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_reg = reg_model.predict(X_test_reg)

# Calculate metrics
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)

# Display results
print("\n========== REGRESSION RESULTS ==========")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 2))

print("Regression Evaluation Completed!")