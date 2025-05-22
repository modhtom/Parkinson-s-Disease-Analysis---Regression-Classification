import pandas as pd
import numpy as np
import pickle
import os
from ast import literal_eval
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)


def convert_to_minutes(x):
    try:
        if pd.isna(x):
            return np.nan
        h, m = map(int, x.split(":"))
        return h * 60 + m
    except:
        return np.nan


def load_artifacts(models_dir):
    artifacts = {}
    files = [
        "scaler.pkl",
        "final_features.pkl",
        "binary_classes.pkl",
        "numeric_cols.pkl",
        "binary_cols.pkl",
        "numeric_medians.pkl",
    ]

    for file in files:
        key = file.replace(".pkl", "")
        with open(os.path.join(models_dir, file), "rb") as f:
            artifacts[key] = pickle.load(f)
    return artifacts


def test_regression(test_path):
    models_dir = "TEST\\models\\REG"
    df = pd.read_csv(test_path)
    artifacts = load_artifacts(models_dir)

    df["EducationLevel"] = df["EducationLevel"].fillna("Unknown")

    df["WeeklyPhysicalActivity_min"] = df["WeeklyPhysicalActivity (hr)"].apply(
        convert_to_minutes
    )

    df.drop("WeeklyPhysicalActivity (hr)", axis=1, inplace=True)
    df.drop(columns=["DoctorInCharge", "PatientID"], inplace=True)

    for col in ["Symptoms", "MedicalHistory"]:
        expanded = pd.json_normalize(df[col].apply(literal_eval))
        expanded = expanded.replace({"Yes": 1, "No": 0})
        df = pd.concat([df.drop(col, axis=1), expanded], axis=1)

    for col, mapping in artifacts["binary_classes"].items():
        df[col] = df[col].map(mapping).fillna(0).astype(int)

    for col in artifacts["numeric_cols"]:
        df[col] = df[col].fillna(artifacts["numeric_medians"][col])

    # df[artifacts["numeric_cols"]] = np.log1p(df[artifacts["numeric_cols"]])
    df[artifacts["numeric_cols"]] = artifacts["scaler"].transform(
        df[artifacts["numeric_cols"]]
    )
    X = df[artifacts["final_features"]]

    models = {}
    for model_file in os.listdir(models_dir):
        if model_file.endswith("_model.pkl"):
            name = model_file[: -len("_model.pkl")]
            with open(os.path.join(models_dir, model_file), "rb") as f:
                models[name] = pickle.load(f)

    results = {}
    for name, model in models.items():
        preds = model.predict(X)
        results[name] = preds

    y_acc = df["UPDRS"]
    for name in models.keys():
        y_pred = results[name]
        print(f"{name} MSE: {mean_squared_error(y_acc, y_pred):.4f}")
        print(f"{name} R2: {r2_score(y_acc, y_pred):.4f}\n")


def test_classification(test_path):
    models_dir = "TEST\\models\\CLS"
    df = pd.read_csv(test_path)

    df["EducationLevel"] = df["EducationLevel"].fillna("Unknown")

    df["WeeklyPhysicalActivity_min"] = df["WeeklyPhysicalActivity (hr)"].apply(
        convert_to_minutes
    )
    df.drop("WeeklyPhysicalActivity (hr)", axis=1, inplace=True)

    df.drop(["DoctorInCharge", "PatientID"], axis=1, errors="ignore", inplace=True)

    for col in ["Symptoms", "MedicalHistory"]:
        if col in df.columns:
            expanded = pd.json_normalize(df[col].apply(literal_eval)).replace(
                {"Yes": 1, "No": 0}
            )
            df = pd.concat([df.drop(col, axis=1), expanded], axis=1)

    mappings = {
        "Smoking": {"Yes": 1, "No": 0},
        "Gender": {"Male": 1, "Female": 0},
        "Ethnicity": {"Caucasian": 0, "Asian": 1, "Other": 2, "African American": 3},
        "EducationLevel": {
            "Unknown": 0,
            "Bachelor's": 2,
            "High School": 1,
            "Higher": 3,
        },
    }
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping).fillna(0).astype(int)

    X = df.drop("Diagnosis", axis=1) if "Diagnosis" in df.columns else df

    df.dropna(inplace=True)
    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]

    models = {}
    for file in os.listdir(models_dir):
        if file.endswith("_model.pkl"):
            name = file[: -len("_model.pkl")]
            with open(os.path.join(models_dir, file), "rb") as f:
                models[name] = pickle.load(f)

    results = {}
    for name, model in models.items():
        results[name] = model.predict(X)

    y_acc = df["Diagnosis"]
    for name in models.keys():
        y_pred = results[name]
        print(f"{name} Accuracy: {accuracy_score(y_acc, y_pred):.4f}")
        print(classification_report(y_acc, y_pred))


type = ""
path = ""
if type == "reg":
    test_regression(path)
elif type == "cls":
    test_classification(path)
