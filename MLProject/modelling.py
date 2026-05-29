import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
    RocCurveDisplay
)

# ============================================================
# MLFLOW SETUP — simpan lokal saja saat di CI
# DagsHub hanya dipakai saat run manual/lokal
# ============================================================
os.environ.pop("MLFLOW_RUN_ID", None)

IS_CI = os.environ.get("GITHUB_ACTIONS") == "true"

if not IS_CI:
    import dagshub
    dagshub.init(
        repo_owner='ChelsYP',
        repo_name='Eksperimen_SML_ChelsaYogaPermadany',
        mlflow=True
    )

mlflow.set_experiment("Raisin_CI_Pipeline")

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('raisin_preprocessing/raisin_preprocessed.csv')

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

print("🔍 Mencari hyperparameter terbaik...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_
print(f"   Best params: {best_params}")

# ============================================================
# EVALUASI
# ============================================================
y_pred       = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_proba)

# ============================================================
# BUAT ARTEFAK
# ============================================================
os.makedirs("artifacts", exist_ok=True)

# Confusion Matrix
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Besni', 'Keceli'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False)
ax.set_title("Confusion Matrix — Random Forest (Tuned)")
plt.tight_layout()
plt.savefig("artifacts/confusion_matrix.png", dpi=150)
plt.close()

# ROC Curve
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax, name="Random Forest")
ax.set_title("ROC Curve — Random Forest (Tuned)")
plt.tight_layout()
plt.savefig("artifacts/roc_curve.png", dpi=150)
plt.close()

# Feature Importance
feat_imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax  = plt.subplots(figsize=(8, 5))
feat_imp.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title("Feature Importance — Random Forest (Tuned)")
ax.set_ylabel("Importance")
plt.tight_layout()
plt.savefig("artifacts/feature_importance.png", dpi=150)
plt.close()

# Classification Report
report = classification_report(y_test, y_pred, target_names=['Besni', 'Keceli'])
with open("artifacts/classification_report.txt", "w") as f:
    f.write(report)

# ============================================================
# MLFLOW LOGGING
# ============================================================
with mlflow.start_run(run_name="RandomForest_CI"):
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(
        best_model,
        "random_forest_tuned",
        registered_model_name="random_forest_tuned"  # daftarkan ke model registry
    )

    mlflow.log_artifact("artifacts/confusion_matrix.png")
    mlflow.log_artifact("artifacts/roc_curve.png")
    mlflow.log_artifact("artifacts/feature_importance.png")
    mlflow.log_artifact("artifacts/classification_report.txt")

    run_id = mlflow.active_run().info.run_id
    print(f"\n✅ Training selesai!")
    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    print(f"   ROC AUC  : {roc_auc:.4f}")
    print(f"   Run ID   : {run_id}")

# Simpan run_id ke file untuk dipakai Docker build
with open("artifacts/run_id.txt", "w") as f:
    f.write(run_id)
print(f"   Run ID disimpan ke artifacts/run_id.txt")