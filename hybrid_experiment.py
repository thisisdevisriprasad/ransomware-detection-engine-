"""
hybrid_experiments.py

Full experiment suite:
- Train PCA+KMeans (unsupervised) and RandomForest (supervised)
- Hybrid ensemble: percentage_rule & threshold_rule (ensemble, not stacking)
- Create perturbed test datasets to simulate "similar" data sources
- Evaluate accuracy & inference time on known vs unknown patterns
- Save results to hybrid_experiment_results.csv

Usage:
    python hybrid_experiments.py

Notes:
- Place data_file.csv and data_file_unlabeled.csv in same folder or adjust paths.
- Requires: pandas, numpy, scikit-learn, joblib
"""

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import joblib
import os

# ---------- Utilities & Helpers ----------

def keep_numeric_and_clean(df):
    """Keep numeric columns and fill missing values with mean."""
    df_num = df.select_dtypes(include=['number']).copy()
    if df_num.shape[1] == 0:
        raise ValueError("No numeric columns found in dataframe.")
    df_num = df_num.fillna(df_num.mean())
    return df_num

def create_perturbed_dataset(X, method="noise", scale=0.1, drop_frac=0.0, seed=42):
    """Create perturbed variation of X to simulate another dataset.
    method:
        - "noise": add gaussian noise (scale fraction of std)
        - "scale": multiply features by random scale near 1.0
        - "drop": set a fraction of features to zero to simulate missing features
        - "mix": combine noise + scale
    """
    rng = np.random.RandomState(seed)
    Xp = X.copy().astype(float)
    if method in ("noise", "mix"):
        stds = Xp.std(axis=0, ddof=0)
        noise = rng.normal(loc=0.0, scale=scale * (stds + 1e-8), size=Xp.shape)
        Xp = Xp + noise
    if method in ("scale", "mix"):
        scales = 1.0 + rng.normal(loc=0.0, scale=scale, size=(1, Xp.shape[1]))
        Xp = Xp * scales
    if method == "drop":
        # drop (zero-out) a random subset of columns
        n_drop = int(np.floor(drop_frac * Xp.shape[1]))
        if n_drop > 0:
            cols = rng.choice(Xp.shape[1], n_drop, replace=False)
            Xp[:, cols] = 0.0
    return Xp

# ---------- Load Data ----------
print("Loading datasets...")
sup_path = "data_file.csv"
unsup_path = "data_file_unlabeled.csv"

if not os.path.exists(sup_path) or not os.path.exists(unsup_path):
    raise FileNotFoundError("Make sure data_file.csv and data_file_unlabeled.csv are in the current folder.")

supervised_df = pd.read_csv(sup_path)
unsupervised_df = pd.read_csv(unsup_path)

print("Supervised shape:", supervised_df.shape)
print("Unsupervised shape:", unsupervised_df.shape)

# ---------- Preprocess ----------
# For unlabeled: keep numeric and clean
unsup_num = keep_numeric_and_clean(unsupervised_df)

# For supervised: find target column (common names or infer)
# Try common names, else infer by low unique values / presence of 0/1
target_candidates = ['target', 'label', 'class', 'y']
target_col = None
for c in target_candidates:
    if c in supervised_df.columns:
        target_col = c
        break

if target_col is None:
    # guess: integer column with only 0/1 (or few uniques)
    for col in supervised_df.select_dtypes(include=['number']).columns:
        uniques = supervised_df[col].dropna().unique()
        if set(np.unique(uniques)).issubset({0,1}) or (len(uniques)<=5 and supervised_df[col].dtype in ['int64','int32']):
            target_col = col
            break

if target_col is None:
    raise ValueError("Could not detect target column. Name your label column as 'target' or 'label' or 'class'.")

print("Detected target column:", target_col)

sup_num = supervised_df.copy()
# drop non-numeric cols from features
X_sup_df = sup_num.drop(columns=[target_col], errors='ignore').select_dtypes(include=['number'])
y_sup = sup_num[target_col].fillna(method='ffill').astype(int)

X_sup_df = X_sup_df.fillna(X_sup_df.mean())

# ---------- Standardize ----------
scaler_unsup = StandardScaler()
X_unsup_scaled = scaler_unsup.fit_transform(unsup_num.values)

scaler_sup = StandardScaler()
X_sup_scaled = scaler_sup.fit_transform(X_sup_df.values)

# ---------- Train Unsupervised Model ----------
print("\nTraining PCA + KMeans (unsupervised)...")
pca = PCA(n_components=2, random_state=42)
X_unsup_pca = pca.fit_transform(X_unsup_scaled)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
unsup_clusters = kmeans.fit_predict(X_unsup_pca)

# We want a cluster->label mapping for when we predict clusters on supervised samples.
# Map each cluster to majority label when applying kmeans to supervised data (projected)
print("Mapping cluster -> class using supervised data (majority mapping)")
X_sup_pca_like = pca.transform(X_sup_scaled)  # project supervised data using same PCA
sup_cluster_pred = kmeans.predict(X_sup_pca_like)

# Map clusters to label by majority vote on training labeled data
cluster_to_label = {}
for c in np.unique(sup_cluster_pred):
    idxs = np.where(sup_cluster_pred == c)[0]
    if len(idxs) == 0:
        cluster_to_label[c] = 0
    else:
        vals = y_sup.iloc[idxs].values
        # majority label
        mapped = int(np.round(vals.mean()))  # works for binary labels 0/1
        cluster_to_label[c] = mapped
print("Cluster->Label mapping:", cluster_to_label)

# Unsupervised classifier: cluster->mapped label
def unsupervised_predict(X_raw):
    # expects already-scaled raw numeric X (same columns as X_sup_df ideally)
    Xp = pca.transform(X_raw)
    clusters = kmeans.predict(Xp)
    mapped = np.array([cluster_to_label[c] for c in clusters])
    return mapped

# ---------- Train Supervised Model ----------
print("\nTraining RandomForest (supervised)...")
X_train, X_test, y_train, y_test = train_test_split(X_sup_scaled, y_sup.values, test_size=0.2, random_state=42, stratify=y_sup)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
sup_acc = accuracy_score(y_test, y_pred)
print("Supervised test accuracy:", sup_acc)

# ---------- Evaluate Unsupervised alone on same labeled test set ----------
unsup_on_test = unsupervised_predict(X_test)
unsup_acc = accuracy_score(y_test, unsup_on_test)
print("Unsupervised-mapped accuracy on supervised test set:", unsup_acc)

# ---------- Hybrid Ensemble Functions ----------
def ensemble_percentage_rule(pred_sup_probs, pred_unsup_label, weight_sup=0.65, weight_unsup=0.35):
    """
    pred_sup_probs: probability vector from supervised model (n_samples, n_classes)
    pred_unsup_label: label predicted by unsupervised mapping (n_samples,)
    This uses weighted probabilities: weighted_sup_probs + weighted_unsup_onehot, then argmax.
    """
    # convert unsupervised label to one-hot
    n_classes = pred_sup_probs.shape[1]
    unsup_onehot = np.zeros_like(pred_sup_probs)
    for i, lab in enumerate(pred_unsup_label):
        if lab < n_classes:
            unsup_onehot[i, int(lab)] = 1.0
        else:
            unsup_onehot[i, 0] = 1.0
    combined = weight_sup * pred_sup_probs + weight_unsup * unsup_onehot
    return combined.argmax(axis=1)

def ensemble_threshold_rule(pred_sup_label, pred_unsup_label, sup_acc_value, threshold=0.85):
    """
    If supervised model accuracy < threshold, fallback to unsupervised (use its label).
    Otherwise use supervised label.
    """
    if sup_acc_value < threshold:
        return pred_unsup_label
    else:
        return pred_sup_label

# ---------- Function to evaluate model on dataset and measure inference time ----------
def evaluate_on_dataset(X_raw_scaled, y_true, mode="percentage", weight_sup=0.65, weight_unsup=0.35, threshold=0.85):
    """
    X_raw_scaled: scaled feature matrix (n_samples, n_features) - same scaling as training
    y_true: true labels or None
    mode: "percentage" or "threshold" or "supervised" or "unsupervised"
    Returns: metrics dict and timing info (avg_latency_ms)
    """
    results = {}
    # get supervised predictions and probabilities
    t0 = time.perf_counter()
    sup_probs = rf.predict_proba(X_raw_scaled)
    t1 = time.perf_counter()
    sup_label = sup_probs.argmax(axis=1)
    time_sup = (t1 - t0)  # time for supervised prob prediction

    # get unsupervised prediction (map clusters)
    t0 = time.perf_counter()
    unsup_label = unsupervised_predict(X_raw_scaled)
    t1 = time.perf_counter()
    time_unsup = (t1 - t0)

    # ensemble decision
    t0 = time.perf_counter()
    if mode == "percentage":
        ensemble_label = ensemble_percentage_rule(sup_probs, unsup_label, weight_sup=weight_sup, weight_unsup=weight_unsup)
    elif mode == "threshold":
        ensemble_label = ensemble_threshold_rule(sup_label, unsup_label, sup_acc_value=sup_acc, threshold=threshold)
    elif mode == "supervised":
        ensemble_label = sup_label
    elif mode == "unsupervised":
        ensemble_label = unsup_label
    else:
        raise ValueError("Unknown ensemble mode")
    t1 = time.perf_counter()
    time_ensemble = (t1 - t0)

    # measure per-sample latency (ms)
    total_time = time_sup + time_unsup + time_ensemble
    avg_latency_ms = (total_time / X_raw_scaled.shape[0]) * 1000.0

    metrics = {}
    if y_true is not None:
        metrics['accuracy'] = accuracy_score(y_true, ensemble_label)
        metrics['precision'] = precision_score(y_true, ensemble_label, zero_division=0)
        metrics['recall'] = recall_score(y_true, ensemble_label, zero_division=0)
        metrics['f1'] = f1_score(y_true, ensemble_label, zero_division=0)
    else:
        metrics['accuracy'] = None
        metrics['precision'] = None
        metrics['recall'] = None
        metrics['f1'] = None

    metrics['avg_latency_ms'] = avg_latency_ms
    metrics['time_sup_total_s'] = time_sup
    metrics['time_unsup_total_s'] = time_unsup
    metrics['time_ensemble_s'] = time_ensemble
    metrics['n_samples'] = X_raw_scaled.shape[0]
    metrics['mode'] = mode
    return metrics, ensemble_label

# ---------- Create variant datasets to test generalization ----------
print("\nGenerating test datasets (perturbations) to simulate other sources...")
# known: the held-out test set from supervised split (X_test, y_test)
X_known = X_test
y_known = y_test

# unknown variants: create perturbations on X_test to simulate domain shift
X_unknown_noise = create_perturbed_dataset(X_test, method="noise", scale=0.2, seed=1)
X_unknown_scale = create_perturbed_dataset(X_test, method="scale", scale=0.15, seed=2)
X_unknown_mix = create_perturbed_dataset(X_test, method="mix", scale=0.18, seed=3)
X_unknown_drop = create_perturbed_dataset(X_test, method="drop", drop_frac=0.2, seed=4)

variants = {
    "known_test": (X_known, y_known),
    "unknown_noise": (X_unknown_noise, y_known),
    "unknown_scale": (X_unknown_scale, y_known),
    "unknown_mix": (X_unknown_mix, y_known),
    "unknown_drop": (X_unknown_drop, y_known),
}

# ---------- Run experiments ----------
results = []
modes = [
    ("supervised", None),
    ("unsupervised", None),
    ("percentage_65_35", (0.65, 0.35)),
    ("percentage_50_50", (0.5, 0.5)),
    ("threshold_85", (0.85,)),  # threshold rule
]

print("\nRunning experiments...")
for variant_name, (Xv, yv) in variants.items():
    for mode_name, params in modes:
        if mode_name == "supervised":
            metrics, _ = evaluate_on_dataset(Xv, yv, mode="supervised")
        elif mode_name == "unsupervised":
            metrics, _ = evaluate_on_dataset(Xv, yv, mode="unsupervised")
        elif mode_name.startswith("percentage"):
            w_sup, w_unsup = params
            metrics, _ = evaluate_on_dataset(Xv, yv, mode="percentage", weight_sup=w_sup, weight_unsup=w_unsup)
        elif mode_name.startswith("threshold"):
            thr = params[0]
            metrics, _ = evaluate_on_dataset(Xv, yv, mode="threshold", threshold=thr)
        else:
            continue
        row = {
            "variant": variant_name,
            "mode": mode_name,
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1": metrics['f1'],
            "avg_latency_ms": metrics['avg_latency_ms'],
            "time_sup_total_s": metrics['time_sup_total_s'],
            "time_unsup_total_s": metrics['time_unsup_total_s'],
            "time_ensemble_s": metrics['time_ensemble_s'],
            "n_samples": metrics['n_samples']
        }
        results.append(row)
        print(f"Variant {variant_name} | Mode {mode_name} | acc {metrics['accuracy']:.4f} | latency {metrics['avg_latency_ms']:.4f} ms")

# Save results
res_df = pd.DataFrame(results)
res_df.to_csv("hybrid_experiment_results.csv", index=False)
print("\nSaved hybrid_experiment_results.csv")

# Save trained models for reuse
joblib.dump(rf, "rf_model.joblib")
joblib.dump(kmeans, "kmeans_model.joblib")
joblib.dump(pca, "pca_model.joblib")
joblib.dump(scaler_sup, "scaler_sup.joblib")
joblib.dump(scaler_unsup, "scaler_unsup.joblib")
print("Saved rf_model.joblib, kmeans_model.joblib, pca_model.joblib, scaler_sup.joblib, scaler_unsup.joblib")

print("\nExperiment complete. Summary:")
print(res_df.groupby('mode')[['accuracy','avg_latency_ms']].mean())
