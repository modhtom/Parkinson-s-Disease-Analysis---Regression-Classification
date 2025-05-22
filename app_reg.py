import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from ast import literal_eval
import scipy.stats as stats
from scipy.stats import kruskal
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LinearRegression
import pickle
import os


# **************************************** (1) Data Loading ****************************************
df = pd.read_csv("parkinsons_disease_data_reg.csv")

print("Initial data shape:", df.shape)
print(df.isnull().sum())

# **************************************** (2) Data Cleaning ****************************************
# fill nulls with categorical value 'Unknown'
df["EducationLevel"] = df["EducationLevel"].fillna("Unknown")


# convert WeeklyPhysicalActivity_min from HH:MM to minuts
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
# drop unneccessary columns 'DoctorInCharge' same value for all rows has no affect
df.drop(columns=["DoctorInCharge", "PatientID"], inplace=True)

# **************************************** (3) Feature Engineering ****************************************
# expand columns have more than one binary(yes/no) value
for col in ["Symptoms", "MedicalHistory"]:
    expanded = pd.json_normalize(df[col].apply(literal_eval))
    expanded = expanded.replace({"Yes": 1, "No": 0})
    df = pd.concat([df.drop(col, axis=1), expanded], axis=1)

# encode categorical columns
binary_classes = {
    "Smoking": {"Yes": 1, "No": 0},
    "Gender": {"Male": 1, "Female": 0},
    "Ethnicity": {"Caucasian": 0, "Asian": 1, "Other": 2, "African American": 3},
    "EducationLevel": {"Unknown": 0, "Bachelor's": 2, "High School": 1, "Higher": 3},
}

for col, mapping in binary_classes.items():
    df[col] = df[col].map(mapping)

# **************************************** (4) Train-Test Split ****************************************
x = df.drop("UPDRS", axis=1)
y = df["UPDRS"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
# on Y make log transformer
# if error not dec make it on all numerical
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)
# **************************************** (5) Feature Scaling ****************************************
# choos only numeric columns for scaling
numeric_cols = x_train.select_dtypes(include=["int64", "float64"]).columns
binary_cols = [col for col in x_train.columns if set(x_train[col].unique()) == {0, 1}]
numeric_cols = list(set(numeric_cols) - set(binary_cols))
print(numeric_cols)
print(binary_cols)
x_train[numeric_cols] = np.log1p(x_train[numeric_cols])
x_test[numeric_cols] = np.log1p(x_test[numeric_cols])

# scaling
# scaler = MinMaxScaler()
scaler = StandardScaler()
# robust scaler
x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])

# **************************************** 6. Feature Selection ****************************************
# correlation on numeric columns
data = df[numeric_cols]
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.05,
    cbar=True,
)
plt.title("Correlation Matrix of All Features", pad=20, fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Kruskal-Wallis(non-parametric ANOVA) on categorical columns
x = df[binary_cols]
y = df["UPDRS"]
for col in x.columns:
    groups = [y[x[col] == cat] for cat in x[col].unique()]
    stat, pval = kruskal(*groups, nan_policy="omit")
    print(f"{col}: Kruskal Stat={stat:.2f}, p={pval:.4f}")


# Lasso regularization
lasso = Lasso(alpha=0.001, random_state=42)
lasso.fit(x_train, y_train)
selected_features_lasso = x_train.columns[lasso.coef_ != 0]
print(selected_features_lasso)
# final features based on (corr matrix, kruskal, lasso)
features_to_keep = set(x_train.columns) - {
    "FunctionalAssessment",
    "CholesterolLDL",
    "Age",
    "WeeklyPhysicalActivity_min",
    "Stroke",
    "TraumaticBrainInjury",
    "FamilyHistoryParkinsons",
    "Hypertension",
    "SpeechProblems",
    "Diabetes",
    "SleepDisorders",
    "Rigidity",
    "Gender",
    "Tremor",
    "Smoking",
    "Bradykinesia",
    "Constipation",
    "Ethnicity",
    "EducationLevel",
}
# features_to_keep.update(selected_features_lasso)

final_features = list(features_to_keep)
print("\nFinal selected features:")
print(final_features)

x_train = x_train[final_features]
x_test = x_test[final_features]

numeric_cols = list(final_features)
x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])


# **************************************** (7) Models Evaluation ****************************************
def evaluate_model(model, model_name, x_train, x_test, y_train, y_test):
    # predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # calculate metrics
    metrics = {
        "Model": model_name,
        "Train MSE": mean_squared_error(y_train, y_train_pred),
        "Test MSE": mean_squared_error(y_test, y_test_pred),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "Train R2": r2_score(y_train, y_train_pred),
        "Test R2": r2_score(y_test, y_test_pred),
    }

    return metrics, y_train_pred, y_test_pred


models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1, random_state=42),
    "Ridge Regression": Ridge(alpha=0.5, random_state=42),
}

all_metrics = {}

# train and evaluate each model
for name, model in models.items():
    # train model
    model.fit(x_train, y_train)

    # evaluate model
    metrics, y_train_pred, y_test_pred = evaluate_model(
        model, name, x_train, x_test, y_train, y_test
    )
    all_metrics[name] = metrics

    # print results
    print(f"\n{name} Evaluation Metrics:")
    for metric_name, value in metrics.items():
        if metric_name != "Model":
            print(f"{metric_name}: {value:.4f}")

metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")
metrics_df.drop(columns=["Model"], inplace=True)

print("\nModel Comparison:")
print(metrics_df)

# **************************************** (8) Models Visualization ****************************************
model_results = {}
for name, model in models.items():
    y_test_pred = model.predict(x_test)
    errors = y_test - y_test_pred
    model_results[name] = {"y_test_pred": y_test_pred, "errors": errors}

    # only for tree-based models, get feature importance
    if hasattr(model, "feature_importances_"):
        model_results[name]["importances"] = pd.Series(
            model.feature_importances_, index=x_train.columns
        )

# a.actual vs predicted plots
plt.figure(figsize=(15, 10))
for i, (name, results) in enumerate(model_results.items(), 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=y_test, y=results["y_test_pred"])
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.title(f"{name}: Actual vs Predicted")
    plt.xlabel("Actual UPDRS")
    plt.ylabel("Predicted UPDRS")
plt.tight_layout()
plt.show()

# b. feature importance
importance_models = {
    name: model
    for name, model in models.items()
    if hasattr(model, "feature_importances_")
}

if importance_models:
    plt.figure(figsize=(15, 5 * len(importance_models)))
    for i, (name, model) in enumerate(importance_models.items(), 1):
        plt.subplot(len(importance_models), 1, i)
        importances = pd.Series(model.feature_importances_, index=x_train.columns)
        importances.nlargest(10).plot(kind="barh")
        plt.title(f"{name}: Top 10 Important Features")
    plt.tight_layout()
    plt.show()

# c.error distribution
plt.figure(figsize=(15, 10))
for i, (name, results) in enumerate(model_results.items(), 1):
    plt.subplot(2, 2, i)
    sns.histplot(results["errors"], kde=True)
    plt.title(f"{name}: Prediction Error Distribution")
    plt.xlabel("Prediction Error")
plt.tight_layout()
plt.show()

# d.residual analysis
plt.figure(figsize=(15, 10))
for i, (name, results) in enumerate(model_results.items(), 1):
    # Residuals vs Predicted
    plt.subplot(2, 2, i)
    sns.residplot(
        x=results["y_test_pred"],
        y=results["errors"],
        lowess=True,
        line_kws={"color": "red"},
    )
    plt.title(f"{name}: Residuals vs Predicted Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

# e.Q-Q Plots
plt.figure(figsize=(15, 10))
for i, (name, results) in enumerate(model_results.items(), 1):
    plt.subplot(2, 2, i)
    stats.probplot(results["errors"], dist="norm", plot=plt)
plt.tight_layout()
plt.show()


# **************************************** (9) Model Save ****************************************
models_dir = "../../TEST/models/REG"

artifacts = {
    "scaler": scaler,
    "final_features": final_features,
    "binary_classes": binary_classes,
    "numeric_cols": numeric_cols,
    "binary_cols": binary_cols,
    "numeric_medians": x_train[numeric_cols].median(),
}

for name, artifact in artifacts.items():
    with open(os.path.join(models_dir, f"{name}.pkl"), "wb") as f:
        pickle.dump(artifact, f)

for name, model in models.items():
    path = os.path.join(models_dir, f"{name}_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name} Test MSE: {mse:.4f}, R2: {r2:.4f}\n")
