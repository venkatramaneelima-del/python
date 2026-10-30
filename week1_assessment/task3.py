import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# PART 1: NumPy Operations
# ==========================================
# Create a 4 students x 3 subjects marks array
marks = np.array([
    [75, 82, 90],
    [60, 65, 70],
    [85, 88, 92],
    [40, 50, 55]
])

# Student averages (axis=1)
student_averages = marks.mean(axis=1)
print("Student averages:", student_averages)

# Subject averages (axis=0)
subject_averages = marks.mean(axis=0)

# Add 5 grace marks by broadcasting
grace_marks = marks + 5

# Compute weighted scores using np.dot
weights = np.array([0.3, 0.3, 0.4])
weighted_scores = np.dot(marks, weights)

# ==========================================
# PART 2: Pandas Data Cleaning & Analysis
# ==========================================
# Load employees.csv using correct relative path
df = pd.read_csv("week1_assessment/employees.csv")

# Info, describe, loc, iloc usage
df.info()
df.describe()
first_row = df.iloc[0]
filtered_df = df[df['salary'] > 40000] if 'salary' in df.columns else df

# Create departments.csv programmatically
dept_data = {
    "department": ["Sales", "HR", "IT", "Finance", "Marketing"],
    "head": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "budget": [500000, 300000, 800000, 600000, 400000]
}
dept_df = pd.DataFrame(dept_data)
dept_df.to_csv("week1_assessment/departments.csv", index=False)

# Introduce missing salaries and a duplicate row for testing cleaning
if 'salary' in df.columns and len(df) >= 2:
    df.loc[0, 'salary'] = np.nan
    df.loc[1, 'salary'] = np.nan
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

# Data Cleaning: drop duplicates, median fill, correct types
df = df.drop_duplicates()
if 'salary' in df.columns:
    df['salary'] = df['salary'].fillna(df['salary'].median())
    df['salary'] = df['salary'].astype(float)

# Groupby for headcount matching expected output format
if 'department' in df.columns and 'salary' in df.columns:
    dept_summary = df.groupby('department').agg(headcount=('salary', 'count'))
    headcount_str = " | ".join([f"{dept} {count}" for dept, count in zip(dept_summary.index, dept_summary['headcount'])])
    print(f"Headcount: {headcount_str}")

    # Merges (inner, left, outer) with departments.csv
    merged_inner = pd.merge(df, dept_df, on='department', how='inner')
    merged_left = pd.merge(df, dept_df, on='department', how='left')
    merged_outer = pd.merge(df, dept_df, on='department', how='outer')
    
    # Identify department with no employees
    dept_full_merge = pd.merge(dept_df, df, on='department', how='left', indicator=True)
    no_emp_dept = dept_full_merge[dept_full_merge['_merge'] == 'left_only']['department'].tolist()
    print(f"Outer join: department with no employees listed -> {no_emp_dept}")

# ==========================================
# PART 3: Matplotlib Charts
# ==========================================
# 1. Bar Chart
plt.figure()
if 'department' in df.columns:
    dept_counts = df['department'].value_counts()
    plt.bar(dept_counts.index, dept_counts.values, color='skyblue')
    plt.title("Headcount per Department")
    plt.xlabel("Department")
    plt.ylabel("Headcount")
    plt.legend(["Headcount"])
    plt.savefig("week1_assessment/bar_chart.png")
