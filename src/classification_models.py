# Classification part: 9 classifiers x 2 tasks (fraud detection, late delivery)
# Each "# %%" block = one notebook cell. Metrics: accuracy, recall, F1.
# Uses the team's shared preprocessing: get_task_splits(df, task, seed=42)
# Run from the repo root:  python src/classification_models.py

# %% 1. Imports and settings
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import load_and_prepare_base, get_task_splits
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42                            # same seed as the team
SLOW_SUBSAMPLE = 20000                       # SVM and kNN train on a subsample
os.makedirs("results", exist_ok=True)

# %% 2. Load data (targets are built inside load_and_prepare_base)
df = load_and_prepare_base()                 # reads data/DataCoSupplyChainDataset.csv
print(df.shape)
print("Fraud rate:", df["target_fraud"].mean().round(4))
print("Late rate :", df["target_late_delivery"].mean().round(4))

# %% 3. Preprocessing
# Handled by get_task_splits in src/preprocessing.py:
# leakage removal, 80/20 stratified split (random_state=42), and
# imputing / scaling / one-hot encoding fitted on the training set only.

# %% 4. Models
def get_models(pos_weight):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "LDA": LinearDiscriminantAnalysis(),
        "Gaussian NB": GaussianNB(),
        "SVM": SVC(class_weight="balanced"),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                class_weight="balanced",
                                                random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, n_jobs=-1,
                                            class_weight="balanced",
                                            random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight,
                                 n_jobs=-1, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced",
                                                random_state=RANDOM_STATE),
    }

# %% 5. Train and evaluate on one task
def run_task(task_name):
    # Shared split: data is already imputed, scaled and encoded
    Xtr_s, Xte_s, ytr, yte = get_task_splits(df, task_name, seed=RANDOM_STATE)
    pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    rows, preds = [], {}
    for name, model in get_models(pos_weight).items():
        t0 = time.time()
        if name in ("SVM", "kNN"):                # too slow on the full set
            idx = np.random.RandomState(RANDOM_STATE).choice(
                len(Xtr_s), min(SLOW_SUBSAMPLE, len(Xtr_s)), replace=False)
            model.fit(Xtr_s[idx], ytr[idx])
        else:
            model.fit(Xtr_s, ytr)
        p = model.predict(Xte_s)
        preds[name] = p
        rows.append({"task": task_name, "model": name,
                     "accuracy": accuracy_score(yte, p),
                     "recall": recall_score(yte, p),
                     "f1": f1_score(yte, p),
                     "train_time_s": round(time.time() - t0, 1)})
        print(f"{task_name:14s} {name:20s} F1={rows[-1]['f1']:.3f}  "
              f"({rows[-1]['train_time_s']}s)")
    return pd.DataFrame(rows), preds, yte

# %% 6. Run both tasks
fraud_res, fraud_preds, fraud_yte = run_task("fraud")
late_res, late_preds, late_yte = run_task("late_delivery")

results = pd.concat([fraud_res, late_res], ignore_index=True)
results.to_csv("results/classification_results.csv", index=False)
print(results.round(4).to_string(index=False))

# %% 7. Comparison charts (one per task)
for task in ["fraud", "late_delivery"]:
    sub = results[results["task"] == task].set_index("model")[["accuracy", "recall", "f1"]]
    sub.plot(kind="bar", figsize=(10, 5))
    plt.title(f"Classifier comparison: {task}")
    plt.ylabel("Score"); plt.ylim(0, 1.05); plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"results/classification_{task}.png", dpi=150)
    plt.show()

# %% 8. Confusion matrix of the best model (by F1) for each task
for task, preds, yte in [("fraud", fraud_preds, fraud_yte),
                         ("late_delivery", late_preds, late_yte)]:
    best = results[results["task"] == task].sort_values("f1", ascending=False).iloc[0]["model"]
    ConfusionMatrixDisplay(confusion_matrix(yte, preds[best])).plot(cmap="Blues")
    plt.title(f"{task}: {best}")
    plt.savefig(f"results/confusion_{task}.png", dpi=150)
    plt.show()

# %% 9. Feature importance check (sanity check for leakage)
# get_task_splits returns arrays without column names, so we print feature indices.
Xtr, Xte, ytr, yte = get_task_splits(df, "late_delivery", seed=RANDOM_STATE)
dt = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE).fit(Xtr, ytr)
top = np.argsort(dt.feature_importances_)[::-1][:10]
for i in top:
    print(f"feature #{i}: importance {dt.feature_importances_[i]:.4f}")
# One feature with importance near 1.0 would suggest leakage.