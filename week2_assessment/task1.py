import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Employee.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

# Step 2: Separate features and target
X = df.drop(columns=["Attrition", "EmployeeID"])
y = df["Attrition"]

print("\nFeatures and Target separated successfully!")

# Step 3: Split 60% training and 40% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# Step 4: Split temporary data into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Step 5: Display results
print("\nTrain Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Test Shape:", X_test.shape)

print("\nData splitting completed successfully!")