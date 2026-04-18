"""
Phase 2: Baseline Model Training & Performance Benchmarking

Goal:
- Establish a reliable baseline performance for the intrusion detection system
  under normal (non-adversarial) conditions.
- Compare different model architectures to understand their strengths and
  inherent limitations prior to adversarial exposure.

Outputs:
- Trained baseline models across three architectures:
  Random Forest, XGBoost, and Multi-Layer Perceptron (MLP).
- Quantitative evaluation metrics on the KDDTest+ dataset, including
  Accuracy, Precision, Recall (Attack class), and F1-score.
- Confusion matrices and summary tables highlighting model behavior and
  identifying intrinsic blind spots (e.g., False Negatives in R2L and U2R attacks).
"""

# ============================================================
# PART 1: LOAD PIPELINE ARTIFACTS FROM PHASE 1
# ============================================================
import numpy as np
import joblib


X_train_scaled = np.load('pipeline/X_train_scaled.npy')
X_test_scaled  = np.load('pipeline/X_test_scaled.npy')

y_train = np.load('pipeline/y_train.npy')
y_test  = np.load('pipeline/y_test.npy')


numerical_mask_bool = joblib.load('pipeline/numerical_mask_bool.pkl')

feature_names_final = joblib.load('pipeline/feature_names_final.pkl')

test_attack_categories = joblib.load('pipeline/test_attack_categories.pkl')

print(" PART 1 COMPLETE — DATA LOADED SUCCESSFULLY")

# ============================================================
# PART 2: RANDOM FOREST CLASSIFIER
# ============================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Training
rf_model.fit(X_train_scaled, y_train)

# Evaluation
y_pred_rf = rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"RF Accuracy: {accuracy_rf*100:.2f}%")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest — Confusion Matrix')
plt.show()

joblib.dump(rf_model, 'pipeline/rf_model.pkl')

print(" PART 2 COMPLETE — RANDOM FOREST")

# ============================================================
# PART 3: XGBOOST CLASSIFIER
# ============================================================

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

normal_count = (y_train == 0).sum()
attack_count = (y_train == 1).sum()
scale_pos_weight = normal_count / attack_count

# Model initialization
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    device='cuda'
)

# Training
xgb_model.fit(X_train_scaled, y_train)

# Evaluation
y_pred_xgb = xgb_model.predict(X_test_scaled)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print(f"XGB Accuracy: {accuracy_xgb*100:.2f}%")
print(classification_report(y_test, y_pred_xgb))

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges')
plt.title('XGBoost — Confusion Matrix')
plt.show()


xgb_model.save_model('pipeline/xgb_model.json')

print(" PART 3 COMPLETE — XGBOOST")

# ============================================================
# PART 4: MLP CLASSIFIER
# ============================================================

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Model initialization
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=100,
    random_state=42,
    verbose=True
)

# Training
mlp_model.fit(X_train_scaled, y_train)

# Evaluation
y_pred_mlp = mlp_model.predict(X_test_scaled)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

print(f"MLP Accuracy: {accuracy_mlp*100:.2f}%")
print(classification_report(y_test, y_pred_mlp))

# Confusion Matrix
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Greens')
plt.title('MLP — Confusion Matrix')
plt.show()


joblib.dump(mlp_model, 'pipeline/mlp_model.pkl')

print(" PART 4 COMPLETE — MLP")

# ============================================================
# PART 5: SUMMARY TABLE & COMPARISON
# ============================================================

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

summary_df = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'MLP'],
    'Accuracy (%)': [
        accuracy_rf * 100,
        accuracy_xgb * 100,
        accuracy_mlp * 100
    ],
    'Recall (Attack)': [
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_xgb),
        recall_score(y_test, y_pred_mlp)
    ],
    'False Negatives': [
        ((y_test == 1) & (y_pred_rf == 0)).sum(),
        ((y_test == 1) & (y_pred_xgb == 0)).sum(),
        ((y_test == 1) & (y_pred_mlp == 0)).sum()
    ]
})

print(summary_df)

summary_df.to_csv('pipeline/phase2_summary.csv', index=False)

print(" PART 5 COMPLETE — PHASE 2 FINISHED")
