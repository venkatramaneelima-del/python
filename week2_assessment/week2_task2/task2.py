import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Step 1: Load dataset
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Check both possible locations
if (BASE_DIR / "Employee.csv").exists():
    DATA_PATH = BASE_DIR / "Employee.csv"
else:
    DATA_PATH = BASE_DIR / "data" / "Employee.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(df.head())
print("Dataset Shape:", df.shape)


# Step 2: Define target
TARGET = "Attrition"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

X = df.drop(columns=[TARGET, "EmployeeID"], errors="ignore")
y = df[TARGET]


# Step 3: Identify column types
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Ordered categorical columns
ordered_candidates = [
    "EducationLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "WorkLifeBalance"
]

ordinal_cols = [
    col for col in categorical_cols
    if col in ordered_candidates
]

onehot_cols = [
    col for col in categorical_cols
    if col not in ordinal_cols
]


# Step 4: Handle missing values
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])


# Step 5: Create ColumnTransformer
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, onehot_cols),
    ("ordinal", ordinal_pipeline, ordinal_cols)
])


# Step 6: Create ML Pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])


# Step 7: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Step 8: Cross Validation
cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=3,
    scoring="accuracy"
)

print("\nCross Validation Accuracy:", cv_scores.mean())


# Step 9: Train pipeline
pipeline.fit(X_train, y_train)

print("\nPipeline trained successfully!")


# Step 10: Test accuracy
accuracy = pipeline.score(X_test, y_test)

print("Test Accuracy:", accuracy)


# Step 11: Save pipeline
joblib.dump(pipeline, BASE_DIR / "employee_pipeline.pkl")

print("\nPipeline saved successfully!")


# Step 12: Load saved pipeline
loaded_pipeline = joblib.load(
    BASE_DIR / "employee_pipeline.pkl"
)

# Predict 3 new raw rows
new_data = X.head(3).copy()

predictions = loaded_pipeline.predict(new_data)

print("\nPredictions for 3 new rows:")
print(predictions)