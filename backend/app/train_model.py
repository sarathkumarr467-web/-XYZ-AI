import os
import json
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

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "student_risk_model.pkl"
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "model_metadata.json"
)


# =========================================================
# CREATE MODEL DIRECTORY
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# START
# =========================================================

print()
print("=" * 60)
print("XYZ AI - STUDENT ML MODEL TRAINING")
print("=" * 60)


# =========================================================
# CHECK DATASET
# =========================================================

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n"
        "Please make sure dataset.csv is inside backend/data/"
    )


# =========================================================
# LOAD DATASET
# =========================================================

print("\n[1/8] Loading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset shape: {df.shape}"
)


# =========================================================
# DISPLAY BASIC INFORMATION
# =========================================================

print("\nDataset columns:")

for index, column in enumerate(df.columns):

    print(
        f"{index + 1}. {column}"
    )


print("\nFirst 5 rows:")

print(
    df.head().to_string()
)


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

print("\n[2/8] Cleaning column names...")

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)


print(
    "\nCleaned columns:"
)

print(
    df.columns.tolist()
)


# =========================================================
# REMOVE COMPLETELY EMPTY COLUMNS
# =========================================================

empty_columns = [
    column
    for column in df.columns
    if df[column].isna().all()
]


if empty_columns:

    print(
        "\nRemoving completely empty columns:"
    )

    for column in empty_columns:

        print(
            f" - {column}"
        )

    df = df.drop(
        columns=empty_columns
    )


# =========================================================
# FIND TARGET COLUMN
# =========================================================

print("\n[3/8] Detecting target column...")


possible_targets = [

    # Student risk
    "risk",
    "risk_level",
    "student_risk",
    "academic_risk",
    "risk_score",
    "risk_category",

    # Performance
    "performance",
    "performance_score",
    "academic_performance",

    # Common ML names
    "target",
    "label",
    "class",
    "outcome",

    # Other common binary targets
    "prediction",
    "result",
]


target_column = None


for target in possible_targets:

    if target in df.columns:

        target_column = target

        break


# =========================================================
# TARGET NOT FOUND
# =========================================================

if target_column is None:

    print()
    print("=" * 60)
    print("TARGET COLUMN NOT FOUND")
    print("=" * 60)

    print(
        "\nAvailable columns:"
    )

    for column in df.columns:

        print(
            f" - {column}"
        )

    print()
    print(
        "The script stopped intentionally."
    )

    print(
        "A wrong target should NOT be selected automatically."
    )

    raise ValueError(
        "\nPlease identify the correct target column "
        "from the columns printed above."
    )


print(
    f"\nTarget column selected: {target_column}"
)


# =========================================================
# REMOVE MISSING TARGET ROWS
# =========================================================

print(
    "\n[4/8] Preparing target..."
)


df = df.dropna(
    subset=[target_column]
).reset_index(
    drop=True
)


if len(df) < 10:

    raise ValueError(
        "Dataset contains fewer than 10 valid rows."
    )


# =========================================================
# FEATURES / TARGET
# =========================================================

X = df.drop(
    columns=[target_column]
)

y = df[target_column]


# =========================================================
# REMOVE ID-LIKE COLUMNS
# =========================================================

columns_to_remove = []


for column in X.columns:

    column_lower = column.lower()

    unique_count = X[column].nunique(
        dropna=True
    )

    # Remove obvious database/index columns
    if column_lower in [
        "id",
        "student_id",
        "index",
        "serial_no",
        "s_no",
    ]:

        columns_to_remove.append(
            column
        )

    # Remove columns that are almost always unique
    elif (
        unique_count == len(X)
        and len(X) > 20
    ):

        columns_to_remove.append(
            column
        )


if columns_to_remove:

    print(
        "\nRemoving ID-like columns:"
    )

    for column in columns_to_remove:

        print(
            f" - {column}"
        )

    X = X.drop(
        columns=columns_to_remove
    )


# =========================================================
# FEATURE TYPES
# =========================================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()


categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


print(
    "\nNumeric features:"
)

print(
    numeric_features
)


print(
    "\nCategorical features:"
)

print(
    categorical_features
)


# =========================================================
# PREPROCESSING
# =========================================================

print(
    "\n[5/8] Creating preprocessing pipeline..."
)


numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )

    ]
)


categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]
)


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )

    ],

    remainder="drop"
)


# =========================================================
# DETERMINE MODEL TYPE
# =========================================================

print(
    "\n[6/8] Selecting ML model..."
)


is_classification = (

    y.dtype == "object"

    or str(y.dtype).startswith(
        "category"
    )

    or y.nunique() <= 10
)


if is_classification:

    model_type = "classification"

    model = RandomForestClassifier(

        n_estimators=200,

        random_state=42,

        class_weight="balanced",

        n_jobs=-1
    )

else:

    model_type = "regression"

    model = RandomForestRegressor(

        n_estimators=200,

        random_state=42,

        n_jobs=-1
    )


print(
    f"Model type: {model_type}"
)


# =========================================================
# COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )

    ]
)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

if is_classification:

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(

                X,

                y,

                test_size=0.20,

                random_state=42,

                stratify=y
            )
        )

    except ValueError:

        print(
            "\nStratified split unavailable."
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(

                X,

                y,

                test_size=0.20,

                random_state=42
            )
        )

else:

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42
        )
    )


print(
    f"\nTraining rows: {len(X_train)}"
)

print(
    f"Testing rows: {len(X_test)}"
)


# =========================================================
# TRAIN MODEL
# =========================================================

print(
    "\n[7/8] Training model..."
)


pipeline.fit(
    X_train,
    y_train
)


print(
    "Training completed successfully."
)


# =========================================================
# PREDICTION
# =========================================================

y_pred = pipeline.predict(
    X_test
)


# =========================================================
# EVALUATION
# =========================================================

print()
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


metrics = {}


if is_classification:

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    metrics = {

        "accuracy": round(
            float(accuracy),
            4
        ),

        "precision": round(
            float(precision),
            4
        ),

        "recall": round(
            float(recall),
            4
        ),

        "f1_score": round(
            float(f1),
            4
        )

    }


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
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )


    metrics = {

        "mae": round(
            float(mae),
            4
        ),

        "mse": round(
            float(mse),
            4
        ),

        "r2": round(
            float(r2),
            4
        )

    }


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

print(
    "\n[8/8] Saving model..."
)


joblib.dump(
    pipeline,
    MODEL_PATH
)


# =========================================================
# SAVE METADATA
# =========================================================

metadata = {

    "dataset": "dataset.csv",

    "target_column": target_column,

    "model_type": model_type,

    "features": X.columns.tolist(),

    "numeric_features": numeric_features,

    "categorical_features": categorical_features,

    "metrics": metrics

}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# =========================================================
# FINAL RESULT
# =========================================================

print()
print("=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print(
    f"\nModel saved to:"
)

print(
    MODEL_PATH
)

print(
    f"\nMetadata saved to:"
)

print(
    METADATA_PATH
)

print(
    "\n✅ ML training completed successfully."
)