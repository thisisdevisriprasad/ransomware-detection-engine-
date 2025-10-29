
#!/usr/bin/env python3
"""
EDA + Supervised Algorithms (classification + regression)
Auto-detects target, runs EDA, preprocesses data, trains multiple models
and saves evaluation results + best model. Designed to run on /mnt/data/data_file.csv.

Outputs (in /mnt/data/eda_supervised_output):
 - quick_eda_summary.csv
 - correlation_matrix.png
 - hist_<feature>.png  (up to 12 features)
 - model_results.csv
 - best_model_<name>.joblib (if training succeeded)
 - confusion_matrix_best.png (classification)

Notes:
 - Uses matplotlib (no seaborn).
 - Tries to use xgboost if installed; otherwise uses sklearn GradientBoosting models.
 - Designed to be robust and reasonably fast: uses sampling for CV and conservative model sizes.
"""
import os, shutil

# Make sure it zips to the same folder as your script
output_dir = os.path.join(os.getcwd(), "eda_supervised_output")
zip_path = os.path.join(os.getcwd(), "eda_supervised_output.zip")

shutil.make_archive(zip_path.replace(".zip", ""), 'zip', output_dir)
print(f"✅ Zipped folder saved at: {zip_path}")

import os, sys, math, warnings, joblib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Optional: XGBoost (if available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Paths
DATA_PATH = "data_file.csv"
OUT_DIR = "/mnt/data/eda_supervised_output"
os.makedirs(OUT_DIR, exist_ok=True)

def load_data(path):
    if not os.path.exists(path):
        print("Data file not found:", path)
        sys.exit(1)
    df = pd.read_csv(path)
    return df

def auto_detect_target(df):
    candidate_names = [c for c in df.columns if c.lower() in ("target","label","class","y","response","outcome","benign","malicious")]
    if candidate_names:
        return candidate_names[0]
    else:
        return df.columns[-1]

def drop_high_cardinality_ids(X, thresh_frac=0.05, thresh_abs=200):
    drop_cols = []
    for c in X.select_dtypes(include=["object","category"]).columns:
        nuniq = X[c].nunique(dropna=True)
        if nuniq > thresh_abs or nuniq > thresh_frac * len(X):
            drop_cols.append(c)
    return X.drop(columns=drop_cols), drop_cols

def save_quick_eda(df, target_col, numeric_feats, categorical_feats, dropped_cols):
    summary = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "target": target_col,
        "task_suggestion": "classification" if (not pd.api.types.is_numeric_dtype(df[target_col]) or df[target_col].nunique()<=20) else "regression",
        "target_unique_values": int(df[target_col].nunique()),
        "numeric_features": int(len(numeric_feats)),
        "categorical_features": int(len(categorical_feats)),
        "dropped_high_cardinality_columns": ", ".join(dropped_cols) if dropped_cols else ""
    }
    out = os.path.join(OUT_DIR, "quick_eda_summary.csv")
    pd.DataFrame([summary]).to_csv(out, index=False)
    print("Saved EDA summary ->", out)

def generate_plots(df, numeric_feats):
    # correlation matrix
    if len(numeric_feats) >= 2:
        try:
            corr = df[numeric_feats].corr()
            plt.figure(figsize=(8,6))
            plt.title("Correlation matrix")
            plt.imshow(corr, aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(numeric_feats)), numeric_feats, rotation=90)
            plt.yticks(range(len(numeric_feats)), numeric_feats)
            plt.tight_layout()
            path = os.path.join(OUT_DIR, "correlation_matrix.png")
            plt.savefig(path); plt.close()
            print("Saved correlation matrix ->", path)
        except Exception as e:
            print("Correlation plot failed:", e)
    # histograms up to 12 numeric features
    for col in numeric_feats[:12]:
        try:
            plt.figure(); plt.title(f"Histogram: {col}"); df[col].hist(); plt.xlabel(col); plt.ylabel("count"); plt.tight_layout()
            p = os.path.join(OUT_DIR, f"hist_{col}.png"); plt.savefig(p); plt.close()
        except Exception as e:
            print("Histogram failed for", col, ":", e)
    # scatter matrix if small number of numeric features
    try:
        if 2 <= len(numeric_feats) <= 8:
            sm = scatter_matrix(df[numeric_feats].dropna().sample(min(500, len(df))), alpha=0.6, figsize=(8,8))
            fig = sm[0,0].get_figure(); fig.suptitle("Scatter matrix (sampled up to 500 rows)"); fig.tight_layout()
            sp = os.path.join(OUT_DIR, "scatter_matrix.png"); fig.savefig(sp); plt.close(fig)
            print("Saved scatter matrix ->", sp)
    except Exception as e:
        print("Scatter matrix failed:", e)

def prepare_preprocessor(X, numeric_feats, categorical_feats):
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    if categorical_feats:
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
        preprocessor = ColumnTransformer(transformers=[("num", num_pipe, numeric_feats), ("cat", cat_pipe, categorical_feats)], remainder="drop", verbose_feature_names_out=False)
    else:
        preprocessor = ColumnTransformer(transformers=[("num", num_pipe, numeric_feats)], remainder="drop", verbose_feature_names_out=False)
    return preprocessor

def build_models(task):
    models = {}
    if task == "classification":
        models.update({
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "SVC": SVC(probability=True),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "GaussianNB": GaussianNB(),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
        })
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100, random_state=42)
    else:
        models.update({
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
            "SVR": SVR(),
            "KNNRegressor": KNeighborsRegressor(n_neighbors=5),
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
        })
        if XGBOOST_AVAILABLE:
            models["XGBoostRegressor"] = xgb.XGBRegressor(n_estimators=100, random_state=42)
    return models

def evaluate_classification(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    results = {}
    results["accuracy"] = float(accuracy_score(y_test, y_pred))
    results["precision_macro"] = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    results["recall_macro"] = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    results["f1_macro"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    # ROC AUC for binary only
    try:
        if len(np.unique(y_test)) == 2 and hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)[:,1]
            results["roc_auc"] = float(roc_auc_score(y_test, proba))
        else:
            results["roc_auc"] = float("nan")
    except Exception:
        results["roc_auc"] = float("nan")
    return results

def evaluate_regression(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results = {
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred))
    }
    return results

def main():
    print("Loading data from", DATA_PATH)
    df = load_data(DATA_PATH)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    target_col = auto_detect_target(df)
    print("Auto-detected target column:", target_col)

    # drop rows missing target
    df = df.dropna(subset=[target_col])
    print("After dropping missing target rows:", df.shape)

    # prepare X, drop high-cardinality id-like columns
    X = df.drop(columns=[target_col])
    X_clean, dropped_cols = drop_high_cardinality_ids(X)
    if dropped_cols:
        print("Dropped high-cardinality columns:", dropped_cols)
    numeric_feats = X_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feats = X_clean.select_dtypes(include=["object","category","bool"]).columns.tolist()

    save_quick_eda(df, target_col, numeric_feats, categorical_feats, dropped_cols)
    generate_plots(df, numeric_feats)

    # Determine task automatically
    y = df[target_col]
    n_unique = y.nunique(dropna=True)
    is_numeric_target = pd.api.types.is_numeric_dtype(y)
    task = "regression" if (is_numeric_target and n_unique > 20) else "classification"
    print(f"Detected task: {task} (unique target values = {n_unique})")

    # Encode target for classification if needed
    label_encoder = None
    if task == "classification" and not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y.astype(str))
    else:
        y_encoded = y.values

    # train-test split (stratify for classification)
    stratify_arg = y_encoded if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_encoded, test_size=0.2, random_state=42, stratify=stratify_arg)
    print("Train/Test shapes:", X_train.shape, X_test.shape)

    preprocessor = prepare_preprocessor(X_clean, numeric_feats, categorical_feats)

    models = build_models(task)
    print("Models to run:", list(models.keys()))

    results = []
    best_score = -1e12
    best_model_name = None
    best_pipeline = None

    # For cross-val, use sampled subset to keep it fast
    max_cv_samples = 3000
    if len(X_clean) > max_cv_samples:
        X_cv = X_clean.sample(max_cv_samples, random_state=42)
        y_cv = y_encoded[X_cv.index]
    else:
        X_cv = X_clean
        y_cv = y_encoded

    for name, estimator in models.items():
        print("\\n--- Running model:", name, "---")
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])

        # cross-validation (3 folds) with limited samples
        try:
            if task == "classification":
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scoring = "accuracy"
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scoring = "r2"
            cv_scores = cross_val_score(pipe, X_cv, y_cv, cv=cv, scoring=scoring, n_jobs=2)
            mean_cv = float(np.mean(cv_scores)); std_cv = float(np.std(cv_scores))
        except Exception as e:
            print("Cross-val failed for", name, ":", e)
            mean_cv, std_cv = float("nan"), float("nan")

        # fit on full train
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            print("Training failed for", name, ":", e)
            continue

        # evaluate on test set
        if task == "classification":
            metrics = evaluate_classification(pipe, X_test, y_test)
            # choose accuracy as primary for selection
            primary_score = metrics.get("accuracy", float("nan"))
            result_row = {"model": name, "task": task, "cv_mean": mean_cv, "cv_std": std_cv}
            result_row.update(metrics)
        else:
            metrics = evaluate_regression(pipe, X_test, y_test)
            primary_score = metrics.get("r2", float("nan"))
            result_row = {"model": name, "task": task, "cv_mean": mean_cv, "cv_std": std_cv}
            result_row.update(metrics)

        results.append(result_row)
        # track best (use primary score; higher is better for both accuracy and r2)
        try:
            if not math.isnan(primary_score) and primary_score > best_score:
                best_score = primary_score
                best_model_name = name
                best_pipeline = pipe
        except Exception:
            pass

        print("Model result:", result_row)

    # Save results table
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(OUT_DIR, "model_results.csv")
    results_df.to_csv(results_csv, index=False)
    print("\\nSaved model results ->", results_csv)

    # Save best model
    if best_pipeline is not None:
        best_path = os.path.join(OUT_DIR, f"best_model_{best_model_name}.joblib")
        try:
            joblib.dump(best_pipeline, best_path)
            print("Saved best model ->", best_path)
        except Exception as e:
            print("Failed to save best model:", e)

        # If classification, produce confusion matrix for best
        if task == "classification":
            try:
                y_pred_best = best_pipeline.predict(X_test)
                cm = confusion_matrix(y_test, y_pred_best)
                plt.figure(figsize=(5,5))
                plt.title(f"Confusion matrix: {best_model_name}")
                plt.imshow(cm, aspect='auto')
                plt.colorbar()
                plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
                cm_path = os.path.join(OUT_DIR, "confusion_matrix_best.png")
                plt.savefig(cm_path); plt.close()
                print("Saved confusion matrix ->", cm_path)
            except Exception as e:
                print("Failed to plot confusion matrix:", e)
    else:
        print("No model succeeded; nothing to save as best_model.")

    print("\\nAll done. Output files in:", OUT_DIR)
    print(os.listdir(OUT_DIR))

# Create a ZIP file of your output folder
shutil.make_archive("/mnt/data/eda_supervised_output", 'zip', "/mnt/data/eda_supervised_output")
print("Zipped -> /mnt/data/eda_supervised_output.zip")


if __name__ == "__main__":
    main()
