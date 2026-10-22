from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data" / "creditcard.csv"
images_dir = project_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

frame = pd.read_csv(data_path)

duplicate_rows = frame.duplicated()
print("Exact duplicate rows removed:", duplicate_rows.sum())

frame = frame.drop_duplicates()

train_df, remaining_df = train_test_split(
    frame, test_size=0.4, random_state=5, stratify=frame["Class"]
)
validation_df, test_df = train_test_split(
    remaining_df,
    test_size=0.5,
    random_state=5,
    stratify=remaining_df["Class"],
)

unique_train=train_df.drop_duplicates()
unique_validation=validation_df.drop_duplicates()

overlap=unique_train.merge(unique_validation,on=frame.columns.tolist(),how="inner")
print("Exact row overlap between training and validation:", len(overlap))

X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]
print("Training features and labels:", X_train.shape, y_train.shape)

X_validation = validation_df.drop(columns=["Class"])
y_validation = validation_df["Class"]
print("Validation features and labels:", X_validation.shape, y_validation.shape)

print("\nBaseline — Always predict normal (validation)")

baseline_model=DummyClassifier(strategy="constant",constant=0)
baseline_model.fit(X_train,y_train)

baseline_predictions=baseline_model.predict(X_validation)

print("Validation transactions:", len(baseline_predictions))

correct=baseline_predictions==y_validation

accuracy=(correct.sum()/len(baseline_predictions))*100
print("Accuracy (%):", accuracy)

missed_fraud=(y_validation==1) & (baseline_predictions==0)
print("Missed fraud cases:", missed_fraud.sum())

print("\nLogistic Regression — Validation, default prediction rule")

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_validation)

predicted_fraud = logistic_predictions == 1
print("Predicted fraud alerts:", predicted_fraud.sum())

caught_fraud = (logistic_predictions == 1) & (y_validation == 1)
print("True Positives (TP):", caught_fraud.sum())

recall=caught_fraud.sum()/(y_validation==1).sum()
print("Recall:", recall)

precision=caught_fraud.sum()/predicted_fraud.sum()
print("Precision:", precision)

false_positive=(logistic_predictions==1)&(y_validation==0)
print("False Positives (FP):", false_positive.sum())

false_negative=(logistic_predictions==0)&(y_validation==1)
print("False Negatives (FN):", false_negative.sum())

true_negative=(logistic_predictions==0)&(y_validation==0)
print("True Negatives (TN):", true_negative.sum())

F1=2*precision*recall/(precision+recall)
print("F1:", F1)

probabilities = logistic_model.predict_proba(X_validation)

fraud_probabilities = probabilities[:, 1]

precisions, recalls, thresholds = precision_recall_curve(
    y_validation,
    fraud_probabilities
)
plt.figure()
plt.plot(recalls, precisions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(
    images_dir / "precision_recall_curve.png",
    dpi=300
)
ap = average_precision_score(y_validation, fraud_probabilities)
print("Average Precision (AP):", ap)

fpr, tpr, roc_thresholds = roc_curve(
    y_validation,
    fraud_probabilities
)
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig(
    images_dir / "roc_curve.png",
    dpi=300
)

roc_auc = roc_auc_score(y_validation, fraud_probabilities)
print("ROC-AUC:", roc_auc)

print("\nBalanced Logistic Regression — Validation, default prediction rule")

balanced_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

balanced_model.fit(X_train, y_train)
balanced_predictions = balanced_model.predict(X_validation)

balanced_tp = ((balanced_predictions == 1) & (y_validation == 1)).sum()
balanced_fp = ((balanced_predictions == 1) & (y_validation == 0)).sum()

print("True Positives (TP):", balanced_tp)
print("False Positives (FP):", balanced_fp)

balanced_recall = balanced_tp / (y_validation == 1).sum()
balanced_precision = balanced_tp / (balanced_tp + balanced_fp)

print("Recall:", balanced_recall)
print("Precision:", balanced_precision)

balanced_f1 = (
    2 * balanced_precision * balanced_recall
    / (balanced_precision + balanced_recall)
)

balanced_probabilities = balanced_model.predict_proba(X_validation)[:, 1]

balanced_ap = average_precision_score(y_validation, balanced_probabilities)
balanced_auc = roc_auc_score(y_validation, balanced_probabilities)

print("F1:", balanced_f1)
print("Average Precision (AP):", balanced_ap)
print("ROC-AUC:", balanced_auc)

print("\nRandom Forest — Validation, default prediction rule")

forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=5,
    n_jobs=-1
)

forest_model.fit(X_train, y_train)

forest_predictions = forest_model.predict(X_validation)

forest_tp = ((forest_predictions == 1) & (y_validation == 1)).sum()
forest_fp = ((forest_predictions == 1) & (y_validation == 0)).sum()

print("True Positives (TP):", forest_tp)
print("False Positives (FP):", forest_fp)

forest_recall = forest_tp / (y_validation == 1).sum()
forest_precision = forest_tp / (forest_tp + forest_fp)

forest_f1 = (
    2 * forest_precision * forest_recall
    / (forest_precision + forest_recall)
)

forest_probabilities = forest_model.predict_proba(X_validation)[:, 1]

forest_ap = average_precision_score(y_validation, forest_probabilities)
forest_auc = roc_auc_score(y_validation, forest_probabilities)

print("Precision:", forest_precision)
print("Recall:", forest_recall)
print("F1:", forest_f1)
print("Average Precision (AP):", forest_ap)
print("ROC-AUC:", forest_auc)

forest_predictions_03 = forest_probabilities >= 0.3

forest_tp_03 = (forest_predictions_03 & (y_validation == 1)).sum()
forest_fp_03 = (forest_predictions_03 & (y_validation == 0)).sum()


forest_precision_03 = forest_tp_03 / (forest_tp_03 + forest_fp_03)
forest_recall_03 = forest_tp_03 / (y_validation == 1).sum()

forest_f1_03 = (
    2 * forest_precision_03 * forest_recall_03
    / (forest_precision_03 + forest_recall_03)
)


model_probabilities = {
    "Logistic Regression": fraud_probabilities,
    "Balanced Logistic Regression": balanced_probabilities,
    "Random Forest": forest_probabilities
}

threshold_candidates = [0.2, 0.3, 0.5]

comparison_results = []

for model_name, probabilities in model_probabilities.items():
    for threshold in threshold_candidates:
        predictions = probabilities >= threshold
        tp = (predictions & (y_validation == 1)).sum()
        fp = (predictions & (y_validation == 0)).sum()
        fn = ((~predictions) & (y_validation == 1)).sum()
        candidate_precision = tp / (tp + fp) if tp + fp > 0 else 0
        candidate_recall = tp / (tp + fn)
        candidate_f1 = 2 * tp / (2 * tp + fp + fn)
        comparison_results.append({
            "Model": model_name,
            "Threshold": threshold,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": candidate_precision,
            "Recall": candidate_recall,
            "F1": candidate_f1
        })

comparison_table = pd.DataFrame(comparison_results)
print("\nValidation model and threshold comparison:")
print(comparison_table.to_string(index=False))

final_model = forest_model
final_threshold = 0.3

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_predictions = test_probabilities >= final_threshold

test_tp = (test_predictions & (y_test == 1)).sum()
test_fp = (test_predictions & (y_test == 0)).sum()
test_fn = ((~test_predictions) & (y_test == 1)).sum()
test_tn = ((~test_predictions) & (y_test == 0)).sum()

print("\nFinal test results — Random Forest, threshold = 0.3")
print("True Positives (TP):", test_tp)
print("False Positives (FP):", test_fp)
print("False Negatives (FN):", test_fn)
print("True Negatives (TN):", test_tn)

test_precision = test_tp / (test_tp + test_fp)
test_recall = test_tp / (test_tp + test_fn)
test_f1 = 2 * test_tp / (2 * test_tp + test_fp + test_fn)

test_ap = average_precision_score(y_test, test_probabilities)
test_auc = roc_auc_score(y_test, test_probabilities)

print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1:", test_f1)
print("Average Precision (AP):", test_ap)
print("ROC-AUC:", test_auc)

plt.show()
