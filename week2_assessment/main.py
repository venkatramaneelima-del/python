from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
# Get project folder location
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "results").mkdir(parents=True, exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset location
DATA_PATH = BASE_DIR / "dataset" / "Trainer_dataset.csv"

print("Project folder:", BASE_DIR)
print("Dataset path:", DATA_PATH)
print("Dataset exists:", DATA_PATH.exists())

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())
else:
    print("\nERROR: Dataset file not found!")
    print("Files inside dataset folder:")
    print(list((BASE_DIR / "dataset").glob("*")))
    print("\nDATASET INFORMATION")
df.info()

print("\nSTATISTICAL SUMMARY")
print(df.describe(include="all"))

print("\nTARGET VALUE COUNTS")
print(df["Attrition"].value_counts())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Classification target
X = df.drop(columns=["Attrition", "Annual_Salary", "Employee_ID"])
y = df["Attrition"]

# Identify columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nPREPROCESSING COMPLETED!")
print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])
print("Numerical columns:", numeric_features)
print("Categorical columns:", categorical_features)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Transform training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Model 1: Logistic Regression
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, lr_pred))
# Logistic Regression Probability

print("\nLOGISTIC REGRESSION PROBABILITIES")

probabilities = lr_pipeline.predict_proba(X_test[:5])

print(probabilities)

print("\nPREDICTED VALUES")

print(lr_pipeline.predict(X_test[:5]))

print("\nCLASS NAMES")

print(lr_pipeline.classes_)

# Model 2: Decision Tree
dt_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

dt_pipeline.fit(X_train, y_train)
dt_pred = dt_pipeline.predict(X_test)

print("\nDECISION TREE")
print("Accuracy:", accuracy_score(y_test, dt_pred))


# Model 3: Random Forest
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        oob_score=True,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

print("\nRANDOM FOREST")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("OOB Score:",
      rf_pipeline.named_steps["classifier"].oob_score_)


# Model 4: XGBoost
xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        eval_metric="logloss",
        early_stopping_rounds=10,
        random_state=42
    ))
])

xgb_pipeline.fit(
    X_train,
    y_train,
    classifier__eval_set=[(X_test_processed, y_test)],
    classifier__verbose=False
)

xgb_pred = xgb_pipeline.predict(X_test)

print("\nXGBOOST")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Best iteration:",
      xgb_pipeline.named_steps["classifier"].best_iteration)
# Logistic Regression Probability

print("\nLOGISTIC REGRESSION PROBABILITIES")

probabilities = lr_pipeline.predict_proba(X_test[:5])

print(probabilities)

print("\nPREDICTED VALUES")

print(lr_pipeline.predict(X_test[:5]))

print("\nCLASS NAMES")

print(lr_pipeline.classes_)
# K-MEANS CLUSTERING

X_processed = preprocessor.fit_transform(X)

inertia = []
silhouette_scores = []

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    labels = kmeans.fit_predict(X_processed)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(
        silhouette_score(X_processed, labels)
    )

# Elbow Plot
plt.figure(figsize=(8, 5))
plt.plot(range(2, 7), inertia, marker='o')
plt.title("K-Means Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.savefig(RESULTS_DIR / "elbow_plot.png")
plt.savefig(RESULTS_DIR / "silhouette_plot.png")
plt.show()

# Silhouette Plot
plt.figure(figsize=(8, 5))
plt.plot(range(2, 7), silhouette_scores, marker='o')
plt.title("K-Means Silhouette Scores")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.savefig(BASE_DIR / "results" / "silhouette_plot.png")
plt.show()

print("\nK-MEANS RESULTS")

for k, score in zip(range(2, 7), silhouette_scores):
    print("Clusters:", k, "Silhouette Score:", score)
    import pandas as pd

kmeans_results = pd.DataFrame({
    "Clusters": range(2, 7),
    "Silhouette_Score": silhouette_scores
})

kmeans_results.to_csv(
    RESULTS_DIR / "kmeans_results.csv",
    index=False
)

print("\nK-Means results saved successfully!")
print(kmeans_results)
# FINAL MODEL COMPARISON

import pandas as pd
from sklearn.metrics import accuracy_score

model_results = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ],
    "Validation Score": [
        accuracy_score(y_test, lr_pipeline.predict(X_test)),
        accuracy_score(y_test, dt_pipeline.predict(X_test)),
        accuracy_score(y_test, rf_pipeline.predict(X_test)),
        accuracy_score(y_test, xgb_pipeline.predict(X_test))
    ]
}

results_df = pd.DataFrame(model_results)

print("\nFINAL MODEL COMPARISON")
print(results_df)

results_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)

print("Model comparison saved successfully!")