import pandas as pd
import numpy as np
import pickle
import os
import time
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# **************************************** (1) Data Loading ****************************************
df = pd.read_csv("parkinsons_disease_data_cls.csv")

print("Initial data shape:", df.shape)
print(df.isnull().sum())

# **************************************** (2) Data Cleaning ****************************************
df["EducationLevel"] = df["EducationLevel"].fillna("Unknown")


def convert_to_minutes(x):
    try:
        if pd.isnull(x):
            return np.nan
        parts = x.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return np.nan


df["WeeklyPhysicalActivity_min"] = df["WeeklyPhysicalActivity (hr)"].apply(
    convert_to_minutes
)
df.drop("WeeklyPhysicalActivity (hr)", axis=1, inplace=True)
df.drop(columns=["DoctorInCharge", "PatientID"], inplace=True)

# **************************************** (3) Feature Engineering ****************************************
for col in ["Symptoms", "MedicalHistory"]:
    expanded = pd.json_normalize(df[col].apply(literal_eval))
    expanded = expanded.replace({"Yes": 1, "No": 0})
    df = pd.concat([df.drop(col, axis=1), expanded], axis=1)

binary_classes = {
    "Smoking": {"Yes": 1, "No": 0},
    "Gender": {"Male": 1, "Female": 0},
    "Ethnicity": {"Caucasian": 0, "Asian": 1, "Other": 2, "African American": 3},
    "EducationLevel": {"Unknown": 0, "Bachelor's": 2, "High School": 1, "Higher": 3},
}

for col, mapping in binary_classes.items():
    df[col] = df[col].map(mapping)

# **************************************** (4) Train-Test Split ****************************************
x = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# **************************************** (5) Models Evaluation ****************************************
models = {}
model_training_times = {}

# Logistic Regression
for C in [0.1, 1.0, 10.0]:
    for p in ["l1", "l2"]:
        solver = "liblinear" if p == "l1" else "lbfgs"
        name = f"LR_C{C}_{p}"
        start_time = time.time()
        model = LogisticRegression(C=C, penalty=p, solver=solver, max_iter=1000)
        model.fit(x_train, y_train)
        model_training_times[name] = time.time() - start_time
        models[name] = model

# Random Forest
for n in [50, 100, 200]:
    for max_depth in [None, 10, 20]:
        name = f"RF_est{n}_d{max_depth}"
        start_time = time.time()
        model = RandomForestClassifier(
            n_estimators=n, max_depth=max_depth, random_state=42, n_jobs=-1
        )
        model.fit(x_train, y_train)
        model_training_times[name] = time.time() - start_time
        models[name] = model

# SVM
for C in [0.1, 1, 10]:
    for kernel in ["linear", "rbf", "poly"]:
        name = f"SVM_C{C}_{kernel}"
        start_time = time.time()
        model = SVC(C=C, kernel=kernel, gamma="scale")
        model.fit(x_train, y_train)
        model_training_times[name] = time.time() - start_time
        models[name] = model

# Decision Tree
for max_depth in [None, 5, 10]:
    name = f"DT_d{max_depth}"
    start_time = time.time()
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(x_train, y_train)
    model_training_times[name] = time.time() - start_time
    models[name] = model

best_models = {}
best_model_training_times = {}

for category in ["LR", "RF", "SVM", "DT"]:
    category_models = {}
    for name, model in models.items():
        if name.startswith(category):
            category_models[name] = model

    best_score = -1
    best_model_name = None

    for name, model in category_models.items():
        pred = model.predict(x_train)
        current_score = accuracy_score(y_train, pred)

        if current_score > best_score:
            best_score = current_score
            best_model_name = name

    if best_model_name:
        best_models[category] = category_models[best_model_name]
        best_model_training_times[category] = model_training_times[best_model_name]
        print(f"Best {category} model: {best_model_name}")

estimators = []
for name, model in best_models.items():
    estimators.append((name, model))

# Voting Classifier
start_time = time.time()
vote = VotingClassifier(estimators=estimators, voting="hard")
vote.fit(x_train, y_train)
vote_train_time = time.time() - start_time

# Stacking Classifier
start_time = time.time()
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(solver="lbfgs", max_iter=1000),
    cv=3,
    n_jobs=-1,
)
stack.fit(x_train, y_train)
stack_train_time = time.time() - start_time

ensemble_models = {"voting": vote, "stacking": stack}
all_models = {**best_models, **ensemble_models}

# **************************************** (6) Model Save & Evaluation ****************************************
models_dir = "/content/models/CLS"
os.makedirs(models_dir, exist_ok=True)

accuracies = {}
test_times = {}

for name, model in all_models.items():
    model_path = os.path.join(models_dir, f"{name}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    start_time = time.time()
    preds = model.predict(x_test)
    test_times[name] = time.time() - start_time

    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"Classification Report for {name}:\n{classification_report(y_test, preds)}")

training_times = {
    **best_model_training_times,
    "voting": vote_train_time,
    "stacking": stack_train_time,
}

# **************************************** (7) Visualization ****************************************
model_names = list(all_models.keys())

# Accuracy Plot
plt.figure(figsize=(12, 6))
plt.bar(model_names, [accuracies[name] for name in model_names], color="skyblue")
plt.title("Model Classification Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# Training Time Plot
plt.figure(figsize=(12, 6))
plt.bar(model_names, [training_times[name] for name in model_names], color="lightgreen")
plt.title("Model Training Times")
plt.ylabel("Time (seconds)")
plt.show()

# Testing Time Plot
plt.figure(figsize=(12, 6))
plt.bar(model_names, [test_times[name] for name in model_names], color="salmon")
plt.title("Model Testing Times")
plt.ylabel("Time (seconds)")
plt.show()
