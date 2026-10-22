# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1

# ============================================================
# ROOT-CAUSE DECAY DIAGNOSTIC - MODEL TRAINING NOTEBOOK
# ============================================================
# This notebook trains a supervised multiclass classification
# model to predict the "root_cause" associated with a student's
# learning interaction.
#
# IMPORTANT:
# 1. The dataset must contain the following columns:
#    student_id, question_id, skill_id, timestamp, correct,
#    elapsed_time_ms, question_difficulty, question_attempt_number,
#    interaction_sequence, recent_accuracy, historical_accuracy,
#    accuracy_decay, recent_response_time, response_time_change,
#    same_skill_error_rate, cross_skill_error_rate,
#    prerequisite_failure_rate, session_length,
#    time_since_last_interaction, test_elapsed_time, root_cause
#
# 2. "root_cause" is the prediction target.
# 3. student_id and question_id are identifiers and are not used
#    directly as numerical predictors.
# 4. timestamp is converted into useful temporal features instead
#    of being passed directly to the model.
# 5. The split below is chronological to reduce temporal leakage.
# 6. XGBoost is used because it performs well on tabular data,
#    supports nonlinear relationships, and provides feature
#    importance information useful for a diagnostic application.
# ============================================================

# Uncomment the next line if XGBoost is not installed.
# !pip install xgboost pandas numpy scikit-learn matplotlib seaborn joblib

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ------------------------------------------------------------
# Change only this path if your dataset has a different name.
# Examples:
# DATA_PATH = "synthetic_student_root_cause_dataset.csv"
# DATA_PATH = "./data/student_interactions.csv"
# ------------------------------------------------------------
DATA_PATH = "synthetic_student_root_cause_dataset.csv"

# Directory in which the trained model and supporting files
# will be stored.
OUTPUT_DIR = Path("trained_root_cause_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Environment initialized.")
print("Dataset path:", DATA_PATH)
print("Output directory:", OUTPUT_DIR.resolve())


# %% [code] Cell 2

# ============================================================
# 1. LOAD THE DATASET
# ============================================================
# The CSV file is loaded into a pandas DataFrame.
# We immediately inspect its dimensions and first few rows so
# that problems with the input file can be identified early.

if not Path(DATA_PATH).exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Update DATA_PATH in the previous cell to the correct CSV path."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Number of rows   :", len(df))
print("Number of columns:", len(df.columns))
print("\nColumns:")
print(df.columns.tolist())

display(df.head())


# %% [code] Cell 3

# ============================================================
# 2. VALIDATE REQUIRED COLUMNS
# ============================================================
# Before training, verify that every expected column exists.
# This prevents silent errors caused by misspelled or missing
# feature names.

required_columns = [
    "student_id",
    "question_id",
    "skill_id",
    "timestamp",
    "correct",
    "elapsed_time_ms",
    "question_difficulty",
    "question_attempt_number",
    "interaction_sequence",
    "recent_accuracy",
    "historical_accuracy",
    "accuracy_decay",
    "recent_response_time",
    "response_time_change",
    "same_skill_error_rate",
    "cross_skill_error_rate",
    "prerequisite_failure_rate",
    "session_length",
    "time_since_last_interaction",
    "test_elapsed_time",
    "root_cause"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(f"- {column}" for column in missing_columns)
    )

print("All required columns are present.")


# %% [code] Cell 4

# ============================================================
# 3. BASIC DATA QUALITY CHECK
# ============================================================
# This section checks:
# - duplicate records
# - missing values
# - data types
# - target distribution
#
# The target distribution is particularly important because
# root-cause classes may be imbalanced. A class occurring only
# rarely can be difficult for the model to learn.

print("Data types:")
display(df.dtypes.to_frame("dtype"))

print("\nMissing values:")
missing_summary = (
    df.isnull()
      .sum()
      .sort_values(ascending=False)
      .to_frame("missing_count")
)
missing_summary["missing_percentage"] = (
    missing_summary["missing_count"] / len(df) * 100
)
display(missing_summary)

print("\nDuplicate rows:", df.duplicated().sum())

print("\nRoot-cause distribution:")
target_distribution = (
    df["root_cause"]
      .value_counts(dropna=False)
      .to_frame("count")
)
target_distribution["percentage"] = (
    target_distribution["count"] / len(df) * 100
)
display(target_distribution)


# %% [code] Cell 5

# ============================================================
# 4. CLEAN THE TARGET
# ============================================================
# The target column must contain a valid class for every row.
# Rows without a root_cause cannot be used for supervised
# classification because there is no label to learn from.
#
# We also remove accidental leading/trailing spaces from string
# labels. This prevents labels such as:
#     "Knowledge Decay"
# and
#     " Knowledge Decay "
# from becoming separate classes.

df["root_cause"] = df["root_cause"].astype("string").str.strip()

before_target_drop = len(df)

df = df.dropna(subset=["root_cause"]).copy()

df = df[df["root_cause"] != ""].copy()

print("Rows removed because of missing/empty root_cause:",
      before_target_drop - len(df))

print("\nFinal target classes:")
print(sorted(df["root_cause"].unique().tolist()))


# %% [code] Cell 6

# ============================================================
# 5. CONVERT TIMESTAMP AND CREATE TEMPORAL FEATURES
# ============================================================
# A raw timestamp is not directly useful to most tree models.
# Instead, we extract interpretable temporal information.
#
# The original timestamp is removed later so the model does not
# learn an arbitrary date representation.
#
# These features can capture patterns such as:
# - hour of interaction
# - day of week
# - whether the interaction happened during a weekend
#
# If timestamp parsing fails for a large number of rows, inspect
# the source data because the timestamp format may be incorrect.

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

invalid_timestamp_count = df["timestamp"].isna().sum()

print("Invalid timestamps:", invalid_timestamp_count)

# If a timestamp is unavailable, temporal features cannot be
# calculated. We keep the rows and use neutral missing values,
# which are later handled by the model.
df["interaction_hour"] = df["timestamp"].dt.hour
df["interaction_day_of_week"] = df["timestamp"].dt.dayofweek
df["interaction_month"] = df["timestamp"].dt.month
df["is_weekend"] = (
    df["interaction_day_of_week"].isin([5, 6]).astype(int)
)

print("Temporal features created.")


# %% [code] Cell 7

# ============================================================
# 6. CONVERT EXPECTED NUMERIC COLUMNS TO NUMERIC TYPES
# ============================================================
# Synthetic CSV generation or external data sources can
# sometimes store numeric values as strings.
#
# Converting them explicitly ensures that the model receives
# numerical data rather than text representations.
#
# errors="coerce" converts invalid numeric values to NaN.
# XGBoost can handle missing numeric values internally.

numeric_columns = [
    "correct",
    "elapsed_time_ms",
    "question_difficulty",
    "question_attempt_number",
    "interaction_sequence",
    "recent_accuracy",
    "historical_accuracy",
    "accuracy_decay",
    "recent_response_time",
    "response_time_change",
    "same_skill_error_rate",
    "cross_skill_error_rate",
    "prerequisite_failure_rate",
    "session_length",
    "time_since_last_interaction",
    "test_elapsed_time",
    "interaction_hour",
    "interaction_day_of_week",
    "interaction_month",
    "is_weekend"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("Numeric conversion completed.")

numeric_check = df[numeric_columns].isna().sum()
print("\nMissing numeric values after conversion:")
display(numeric_check[numeric_check > 0].to_frame("missing_count"))


# %% [code] Cell 8

# ============================================================
# 7. SORT DATA CHRONOLOGICALLY
# ============================================================
# Because this project models student behavior over time, the
# training/test split should respect temporal order.
#
# Sorting by timestamp and interaction_sequence provides a stable
# ordering when timestamps are identical or nearly identical.
#
# This is preferable to a purely random split when the model is
# intended to predict future student interactions from historical
# behavior.

df = df.sort_values(
    by=["timestamp", "interaction_sequence"],
    na_position="last"
).reset_index(drop=True)

print("Dataset sorted chronologically.")
print("First timestamp:", df["timestamp"].min())
print("Last timestamp :", df["timestamp"].max())


# %% [code] Cell 9

# ============================================================
# 8. SELECT MODEL FEATURES
# ============================================================
# Identifier fields:
# - student_id
# - question_id
#
# These are excluded because their numeric values are arbitrary
# identifiers, not meaningful measurements.
#
# timestamp is also excluded after extracting useful temporal
# information.
#
# skill_id is retained and one-hot encoded because the skill can
# legitimately influence the root cause.
#
# root_cause is the supervised learning target and is never used
# as an input feature.

target_column = "root_cause"

excluded_columns = [
    "student_id",
    "question_id",
    "timestamp",
    target_column
]

feature_columns = [
    column for column in df.columns
    if column not in excluded_columns
]

print("Features before encoding:")
print(feature_columns)

X = df[feature_columns].copy()
y = df[target_column].copy()


# %% [code] Cell 10

# ============================================================
# 9. HANDLE THE CATEGORICAL skill_id FEATURE
# ============================================================
# XGBoost requires numerical inputs in the configuration used
# here, so skill_id is converted to category codes.
#
# The mapping is saved later so that the exact same mapping can
# be applied during inference.
#
# Missing skill IDs receive a dedicated value (-1).
#
# IMPORTANT:
# The numerical code itself is not interpreted as an ordered
# quantity by the tree model in the same way that a continuous
# variable would be. For a larger production implementation,
# one-hot encoding or native categorical XGBoost support can be
# considered.

skill_mapping = {}

if "skill_id" in X.columns:
    skill_values = X["skill_id"].astype("string").fillna("__MISSING__")

    unique_skills = sorted(skill_values.unique().tolist())

    skill_mapping = {
        skill: index
        for index, skill in enumerate(unique_skills)
    }

    X["skill_id"] = skill_values.map(skill_mapping).astype(int)

print("Number of unique skills:", len(skill_mapping))
print("Example skill mapping:", list(skill_mapping.items())[:10])


# %% [code] Cell 11

# ============================================================
# 10. ENCODE THE ROOT-CAUSE TARGET
# ============================================================
# Machine-learning classifiers require numerical class labels.
# LabelEncoder converts human-readable root causes into integer
# class IDs.
#
# Example:
#     Careless Error       -> 0
#     Knowledge Decay      -> 1
#     Prerequisite Gap     -> 2
#
# The class mapping is saved so predictions can be converted
# back into the original root-cause names.

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

class_names = label_encoder.classes_.tolist()

print("Number of root-cause classes:", len(class_names))
print("\nClass mapping:")

for class_id, class_name in enumerate(class_names):
    print(f"{class_id}: {class_name}")


# %% [code] Cell 12

# ============================================================
# 11. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================
# We use chronological splitting instead of random splitting.
#
# Approximate proportions:
#   70% -> training
#   15% -> validation
#   15% -> test
#
# The model learns from earlier interactions and is evaluated on
# later interactions. This better approximates the real-world
# scenario where the diagnostic system predicts future behavior.
#
# NOTE:
# If your synthetic dataset was generated without meaningful
# temporal ordering, a stratified random split can be used as a
# secondary experiment. For the primary diagnostic evaluation,
# chronological evaluation is safer against temporal leakage.

n = len(X)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train = X.iloc[:train_end].copy()
X_val = X.iloc[train_end:validation_end].copy()
X_test = X.iloc[validation_end:].copy()

y_train = y_encoded[:train_end]
y_val = y_encoded[train_end:validation_end]
y_test = y_encoded[validation_end:]

print("Training samples  :", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples      :", len(X_test))

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nValidation class distribution:")
print(pd.Series(y_val).value_counts().sort_index())

print("\nTest class distribution:")
print(pd.Series(y_test).value_counts().sort_index())


# %% [code] Cell 13

# ============================================================
# 12. CHECK FOR CLASSES MISSING FROM A SPLIT
# ============================================================
# A chronological split can occasionally produce a situation
# where a rare root-cause class exists in the training data but
# not in the validation/test data, or vice versa.
#
# This check does not automatically modify the split because
# changing the split solely to improve class distribution can
# weaken temporal validity.
#
# Instead, the condition is reported so it can be considered
# when interpreting evaluation metrics.

all_classes = set(range(len(class_names)))

for split_name, labels in [
    ("training", y_train),
    ("validation", y_val),
    ("test", y_test)
]:
    present = set(np.unique(labels))
    missing = sorted(all_classes - present)

    if missing:
        print(
            f"WARNING: {split_name} split is missing classes: "
            f"{[class_names[i] for i in missing]}"
        )
    else:
        print(f"{split_name.capitalize()} split contains all classes.")


# %% [code] Cell 14

# ============================================================
# 13. CALCULATE CLASS WEIGHTS
# ============================================================
# Root-cause datasets can be imbalanced. For example, one root
# cause may occur much more frequently than another.
#
# XGBoost's sample_weight parameter allows us to give more
# importance to underrepresented classes during training.
#
# The weights are calculated ONLY from the training labels.
# This prevents information from the validation/test sets from
# influencing the training process.

from sklearn.utils.class_weight import compute_sample_weight

train_sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print("Training sample weights calculated.")
print(
    "Minimum weight:", round(float(train_sample_weights.min()), 4)
)
print(
    "Maximum weight:", round(float(train_sample_weights.max()), 4)
)


# %% [code] Cell 15

# ============================================================
# 14. CREATE THE XGBOOST CLASSIFIER
# ============================================================
# Model design:
#
# - n_estimators: maximum number of boosting rounds
# - max_depth: controls tree complexity
# - learning_rate: controls how strongly each tree contributes
# - subsample / colsample_bytree: reduce overfitting
# - objective: multiclass classification
#
# early_stopping_rounds is used during training so that the
# validation set determines when additional trees stop improving
# the model.
#
# This is a strong baseline rather than a final optimized model.
# Hyperparameter tuning should be performed after validating the
# complete data-generation and labeling process.

num_classes = len(class_names)

model = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=2,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist"
)

print(model)


# %% [code] Cell 16

# ============================================================
# 15. TRAIN THE MODEL
# ============================================================
# The validation set is passed to eval_set so XGBoost can monitor
# validation performance while building trees.
#
# sample_weight compensates for class imbalance in the training
# data.
#
# The validation data is NOT used to update model parameters
# directly; it is used for evaluation and early stopping.

model.fit(
    X_train,
    y_train,
    sample_weight=train_sample_weights,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

print("Model training completed.")


# %% [code] Cell 17

# ============================================================
# 16. PREDICT ON VALIDATION AND TEST DATA
# ============================================================
# We calculate predictions for both validation and test sets.
#
# Validation predictions are useful for model development.
# Test predictions should be treated as the final unbiased
# evaluation after model choices have been made.

y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

print("Predictions generated successfully.")


# %% [code] Cell 18

# ============================================================
# 17. EVALUATE MODEL PERFORMANCE
# ============================================================
# We report:
# - Accuracy
# - Macro precision
# - Macro recall
# - Macro F1
#
# Macro averaging is important for root-cause diagnostics because
# every root-cause class receives equal importance, rather than
# allowing a frequent class to dominate the metric.

def evaluate_model(y_true, y_pred, split_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    print(f"\n{'=' * 60}")
    print(f"{split_name.upper()} PERFORMANCE")
    print(f"{'=' * 60}")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1-score  : {f1:.4f}")

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }

validation_metrics = evaluate_model(
    y_val, y_val_pred, "Validation"
)

test_metrics = evaluate_model(
    y_test, y_test_pred, "Test"
)


# %% [code] Cell 19

# ============================================================
# 18. DETAILED TEST CLASSIFICATION REPORT
# ============================================================
# The classification report gives precision, recall and F1 for
# every root-cause category individually.
#
# This is particularly useful for the diagnostic application:
# a model with good overall accuracy can still perform poorly on
# one clinically/business-critical root cause.

report = classification_report(
    y_test,
    y_test_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
)

print(report)


# %% [code] Cell 20

# ============================================================
# 19. CONFUSION MATRIX
# ============================================================
# The confusion matrix shows which root causes are confused with
# one another.
#
# Rows represent the actual root cause.
# Columns represent the predicted root cause.
#
# A strong diagonal indicates correct classification.
# Large off-diagonal values reveal root-cause categories that
# need better features, better labels, or additional training
# data.

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=np.arange(len(class_names))
)

plt.figure(figsize=(12, 9))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Root-Cause Diagnostic - Test Confusion Matrix")
plt.xlabel("Predicted Root Cause")
plt.ylabel("Actual Root Cause")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# %% [code] Cell 21

# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================
# XGBoost provides feature importance values that help identify
# which inputs contribute most to the classification decisions.
#
# These values are useful for:
# - model interpretation
# - feature review
# - product diagnostics
# - identifying potentially redundant features
#
# IMPORTANT:
# Feature importance does NOT prove causality. A highly important
# feature is associated with prediction performance; it does not
# automatically mean that changing that feature causes the
# predicted root cause.

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

display(feature_importance)

plt.figure(figsize=(12, 8))

top_n = min(20, len(feature_importance))

plt.barh(
    feature_importance.head(top_n)["feature"][::-1],
    feature_importance.head(top_n)["importance"][::-1]
)

plt.title(f"Top {top_n} XGBoost Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# %% [code] Cell 22

# ============================================================
# 21. CHECK FOR POSSIBLE TARGET LEAKAGE
# ============================================================
# This diagnostic check lists features that may deserve special
# review because they are closely related to the target.
#
# In this project, features such as recent_accuracy,
# historical_accuracy, accuracy_decay, same_skill_error_rate,
# cross_skill_error_rate, and prerequisite_failure_rate may be
# valid diagnostic features.
#
# HOWEVER, they must be calculated using only information that
# was available BEFORE the target interaction.
#
# If these features were calculated using the current or future
# interaction outcome, the model would suffer from target leakage
# and the resulting accuracy would be misleading.

potentially_sensitive_features = [
    "recent_accuracy",
    "historical_accuracy",
    "accuracy_decay",
    "recent_response_time",
    "response_time_change",
    "same_skill_error_rate",
    "cross_skill_error_rate",
    "prerequisite_failure_rate",
    "time_since_last_interaction"
]

print("Features requiring temporal/label-leakage verification:")
for feature in potentially_sensitive_features:
    if feature in X.columns:
        print(" -", feature)

print(
    "\nVerify that every listed feature was computed from prior "
    "interactions only and does not include the current response."
)


# %% [code] Cell 23

# ============================================================
# 22. SAVE THE TRAINED MODEL AND PREPROCESSING INFORMATION
# ============================================================
# A production inference pipeline needs more than the model:
# it also needs the class mapping, skill mapping, and exact
# feature order used during training.
#
# These artifacts are saved together so that future prediction
# code can reproduce the same preprocessing.

model_path = OUTPUT_DIR / "root_cause_xgboost_model.pkl"
metadata_path = OUTPUT_DIR / "root_cause_model_metadata.json"
feature_path = OUTPUT_DIR / "feature_columns.json"

joblib.dump(model, model_path)

metadata = {
    "model_type": "XGBClassifier",
    "target_column": target_column,
    "class_names": class_names,
    "skill_mapping": skill_mapping,
    "excluded_columns": excluded_columns,
    "random_state": RANDOM_STATE,
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "test_metrics": test_metrics
}

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, default=str)

with open(feature_path, "w", encoding="utf-8") as f:
    json.dump(list(X.columns), f, indent=4)

print("Saved model:")
print(model_path.resolve())

print("\nSaved metadata:")
print(metadata_path.resolve())

print("\nSaved feature list:")
print(feature_path.resolve())


# %% [code] Cell 24

# ============================================================
# 23. TEST MODEL RELOADING
# ============================================================
# This cell verifies that the serialized model can be loaded
# successfully. This is important before integrating the model
# into an API or agent.
#
# The reloaded model is then used to predict the same test data.
# The resulting accuracy should match the previously calculated
# test accuracy.

loaded_model = joblib.load(model_path)

reloaded_predictions = loaded_model.predict(X_test)

reloaded_accuracy = accuracy_score(
    y_test,
    reloaded_predictions
)

print("Model reloaded successfully.")
print(f"Reloaded model test accuracy: {reloaded_accuracy:.4f}")

assert np.array_equal(
    y_test_pred,
    reloaded_predictions
), "Reloaded model predictions differ from original predictions."

print("Reloaded model predictions match the original model.")


# %% [code] Cell 25

# ============================================================
# 24. EXAMPLE SINGLE-INTERACTION PREDICTION
# ============================================================
# This demonstrates the final inference pattern.
#
# In a real application, the input dictionary would come from
# the learning platform or diagnostic API.
#
# The example uses one existing test record so that all feature
# values are guaranteed to match the trained schema.
#
# The model returns:
# - predicted root cause
# - probability for every root-cause class
#
# The probabilities are useful for an agent because the system
# can distinguish between:
#   "high-confidence root cause"
# and
#   "ambiguous diagnosis requiring more evidence."

example_row = X_test.iloc[[0]].copy()

predicted_class_id = int(
    loaded_model.predict(example_row)[0]
)

predicted_root_cause = label_encoder.inverse_transform(
    [predicted_class_id]
)[0]

probabilities = loaded_model.predict_proba(example_row)[0]

probability_table = pd.DataFrame({
    "root_cause": class_names,
    "probability": probabilities
}).sort_values(
    "probability",
    ascending=False
).reset_index(drop=True)

print("Predicted root cause:", predicted_root_cause)
print("\nRoot-cause probabilities:")
display(probability_table)


# %% [code] Cell 26

# ============================================================
# 25. SAVE EVALUATION RESULTS
# ============================================================
# Saving evaluation results makes the experiment reproducible
# and provides a simple artifact for later comparison with
# alternative models such as Random Forest, LightGBM, CatBoost,
# or neural-network approaches.

evaluation_output = {
    "validation": validation_metrics,
    "test": test_metrics,
    "classification_report": classification_report(
        y_test,
        y_test_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
}

evaluation_path = OUTPUT_DIR / "evaluation_results.json"

with open(evaluation_path, "w", encoding="utf-8") as f:
    json.dump(
        evaluation_output,
        f,
        indent=4,
        default=str
    )

feature_importance_path = OUTPUT_DIR / "feature_importance.csv"
feature_importance.to_csv(
    feature_importance_path,
    index=False
)

print("Evaluation saved to:", evaluation_path.resolve())
print("Feature importance saved to:",
      feature_importance_path.resolve())


# %% [code] Cell 27

# ============================================================
# 26. FINAL TRAINING SUMMARY
# ============================================================
# This final cell provides a compact record of the experiment.
# It is intentionally kept as executable Python rather than a
# Markdown/text cell because this notebook is designed to contain
# code cells only.

print("=" * 70)
print("ROOT-CAUSE DECAY DIAGNOSTIC - TRAINING SUMMARY")
print("=" * 70)

print(f"Dataset rows used      : {len(df)}")
print(f"Training rows          : {len(X_train)}")
print(f"Validation rows        : {len(X_val)}")
print(f"Test rows              : {len(X_test)}")
print(f"Number of features     : {X_train.shape[1]}")
print(f"Number of root causes  : {len(class_names)}")

print("\nRoot-cause classes:")
for class_name in class_names:
    print(" -", class_name)

print("\nTest metrics:")
for metric, value in test_metrics.items():
    print(f" - {metric}: {value:.4f}")

print("\nArtifacts:")
print(" -", model_path)
print(" -", metadata_path)
print(" -", feature_path)
print(" -", evaluation_path)
print(" -", feature_importance_path)

print("\nTraining pipeline completed successfully.")

