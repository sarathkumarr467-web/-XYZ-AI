import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset.csv",
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "student_risk_model.pkl",
)


# =========================================================
# CREATE MODEL DIRECTORY
# =========================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# LOAD DATASET
# =========================================================

print("\n========================================")
print("XYZ AI - ML MODEL TRAINING")
print("========================================")

print("\nLoading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


print("\nCleaned columns:")
print(df.columns.tolist())


# =========================================================
# FIND TARGET COLUMN
# =========================================================

possible_targets = [
    "target",
    "risk",
    "risk_level",
    "student_risk",
    "risk_score",
    "performance",
    "performance_score",
    "hired",
    "label",
    "class",
    "outcome",
]


target_column = None

for column in possible_targets:
    if column in df.columns:
        target_column = column
        break


# =========================================================
# TARGET FALLBACK
# =========================================================

if target_column is None:

    print("\nWARNING:")
    print("Could not automatically identify target column.")

    print("\nAvailable columns:")

    for index, column in enumerate(df.columns):
        print(index, "->", column)

    raise ValueError(
        "\nTarget column could not be identified. "
        "Please check dataset columns."
    )


print(
    f"\nTarget column selected: {target_column}"
)


# =========================================================
# REMOVE EMPTY TARGET ROWS
# =========================================================

df = df.dropna(
    subset=[target_column]
).reset_index(drop=True)


if len(df) < 10:
    raise ValueError(
        "Dataset has too few valid rows after removing "
        "missing target values."
    )


# =========================================================
# FEATURES / TARGET
# =========================================================

X = df.drop(
    columns=[target_column]
)

y = df[target_column]


# =========================================================
# REMOVE COMPLETELY EMPTY COLUMNS
# =========================================================

empty_columns = [
    column
    for column in X.columns
    if X[column].isna().all()
]

if empty_columns:

    print("\nRemoving completely empty columns:")

    for column in empty_columns:
        print("-", column)

    X = X.drop(
        columns=empty_columns
    )


# =========================================================
# REMOVE DUPLICATE COLUMNS
# =========================================================

X = X.loc[
    :,
    ~X.columns.duplicated()
]


# =========================================================
# IDENTIFY DATA TYPES
# =========================================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()


categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


print("\nNumeric features:")
print(numeric_features)


print("\nCategorical features:")
print(categorical_features)


# =========================================================
# PREPROCESSING
# =========================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ],
    remainder="drop",
)


# =========================================================
# DETERMINE MODEL TYPE
# =========================================================

if (
    y.dtype == "object"
    or str(y.dtype).startswith("category")
    or y.nunique() <= 10
):

    model_type = "classification"

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

else:

    model_type = "regression"

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )


print(
    f"\nModel type: {model_type}"
)


# =========================================================
# COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            model,
        ),
    ]
)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

if model_type == "classification":

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )
        )

    except ValueError:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
            )
        )

else:

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )


print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# =========================================================
# TRAIN
# =========================================================

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train,
)


print("Training completed successfully.")


# =========================================================
# PREDICTION
# =========================================================

y_pred = pipeline.predict(
    X_test
)


# =========================================================
# EVALUATION
# =========================================================

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")


if model_type == "classification":

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

else:

    mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    mse = mean_squared_error(
        y_test,
        y_pred,
    )

    r2 = r2_score(
        y_test,
        y_pred,
    )

    print(
        f"MAE : {mae:.4f}"
    )

    print(
        f"MSE : {mse:.4f}"
    )

    print(
        f"R2  : {r2:.4f}"
    )


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    pipeline,
    MODEL_PATH,
)


# =========================================================
# SAVE METADATA
# =========================================================

metadata = {
    "target_column": target_column,
    "model_type": model_type,
    "features": X.columns.tolist(),
}


METADATA_PATH = os.path.join(
    MODEL_DIR,
    "model_metadata.pkl",
)


joblib.dump(
    metadata,
    METADATA_PATH,
)


# =========================================================
# COMPLETE
# =========================================================

print("\n========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")

print(
    f"\nModel:"
)

print(
    MODEL_PATH
)

print(
    "\nMetadata:"
)

print(
    METADATA_PATH
)

print(
    "\nTraining completed successfully! ✅"
)