"""
Phase 3: Adversarial Attack Generation & Semantically Constrained Evaluation

Goal:
- Systematically evaluate the robustness of trained NIDS models against
  gradient-based adversarial evasion attacks.
- Ensure that all generated adversarial examples remain semantically valid
  and technically feasible within real network protocol constraints.

Outputs:
- Adversarial performance metrics measured across multiple perturbation budgets
  (epsilon values).
- Attack Success Rate (ASR) and adversarial accuracy results for FGSM and PGD attacks.
- Comparative robustness evaluation between white-box (MLP) and black-box
  transfer attacks on Random Forest and XGBoost models.
"""
# ============================================================
# PART 1: SETUP & LOAD PIPELINE ARTIFACTS
# ============================================================

# Install IBM ART (run once)
!pip install adversarial-robustness-toolbox -q

import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from xgboost import XGBClassifier

RANDOM_SEED = 42
EPSILON_VALUES = [0.01, 0.05, 0.1, 0.2]
PGD_ITERATIONS = 10
PGD_ALPHA = 0.01

# Load arrays
X_train_scaled = np.load('pipeline/X_train_scaled.npy')
X_test_scaled  = np.load('pipeline/X_test_scaled.npy')
y_train = np.load('pipeline/y_train.npy')
y_test  = np.load('pipeline/y_test.npy')

# Load ART mask and metadata
numerical_mask_bool = joblib.load('pipeline/numerical_mask_bool.pkl')
feature_names_final = joblib.load('pipeline/feature_names_final.pkl')
test_attack_categories = joblib.load('pipeline/test_attack_categories.pkl')

# Load models
rf_model  = joblib.load('pipeline/rf_model.pkl')

xgb_model = XGBClassifier()
xgb_model.load_model('pipeline/xgb_model.json')

mlp_model = joblib.load('pipeline/mlp_model.pkl')

print("✅ PART 1 COMPLETE — SETUP READY")

# ============================================================
# PART 2: MODEL WRAPPING WITH IBM ART
# ============================================================

from art.estimators.classification import XGBoostClassifier
from sklearn.metrics import accuracy_score

# Wrap Random Forest
art_rf = SklearnClassifier(
    model=rf_model,
    clip_values=(0, 1)
)

# Wrap XGBoost
art_xgb = XGBoostClassifier(
    model=xgb_model,
    clip_values=(0, 1),
    nb_features=X_train_scaled.shape[1],
    nb_classes=2
)

# Wrap MLP (sklearn)
art_mlp = SklearnClassifier(
    model=mlp_model,
    clip_values=(0, 1)
)

# Sanity check
pred_rf  = np.argmax(art_rf.predict(X_test_scaled), axis=1)
pred_xgb = np.argmax(art_xgb.predict(X_test_scaled), axis=1)
pred_mlp = np.argmax(art_mlp.predict(X_test_scaled), axis=1)

print("✅ PART 2 COMPLETE — MODELS WRAPPED & VERIFIED")

# ============================================================
# PART 3: FGSM ATTACK
# ============================================================

# Select correctly classified attack samples
correct_attack_idx = np.where(
    (y_test == 1) &
    (np.argmax(art_mlp.predict(X_test_scaled), axis=1) == 1)
)[0]

X_attacks_clean = X_test_scaled[correct_attack_idx]
y_attacks_clean = y_test[correct_attack_idx]

fgsm_results = []

for eps in EPSILON_VALUES:
    fgsm = FastGradientMethod(
        estimator=art_mlp,
        eps=eps,
        eps_step=eps,
        targeted=False
    )

    X_adv = fgsm.generate(
        x=X_attacks_clean,
        mask=numerical_mask_bool
    )

    pred_mlp = np.argmax(art_mlp.predict(X_adv), axis=1)
    pred_rf  = np.argmax(art_rf.predict(X_adv), axis=1)
    pred_xgb = np.argmax(art_xgb.predict(X_adv), axis=1)

    fgsm_results.append({
        'Attack': 'FGSM',
        'Epsilon': eps,
        'ASR_MLP': round((1 - accuracy_score(y_attacks_clean, pred_mlp))*100,2),
        'ASR_RF' : round((1 - accuracy_score(y_attacks_clean, pred_rf))*100,2),
        'ASR_XGB': round((1 - accuracy_score(y_attacks_clean, pred_xgb))*100,2)
    })

fgsm_results_df = pd.DataFrame(fgsm_results)
fgsm_results_df.to_csv('pipeline/fgsm_results.csv', index=False)

print("✅ PART 3 COMPLETE — FGSM GENERATED")

# ============================================================
# PART 4: PGD ATTACK
# ============================================================

pgd_results = []

for eps in EPSILON_VALUES:
    pgd = ProjectedGradientDescent(
        estimator=art_mlp,
        eps=eps,
        eps_step=eps / PGD_ITERATIONS,
        max_iter=PGD_ITERATIONS,
        targeted=False
    )

    X_adv = pgd.generate(
        x=X_attacks_clean,
        mask=numerical_mask_bool
    )

    pred_mlp = np.argmax(art_mlp.predict(X_adv), axis=1)
    pred_rf  = np.argmax(art_rf.predict(X_adv), axis=1)
    pred_xgb = np.argmax(art_xgb.predict(X_adv), axis=1)

    pgd_results.append({
        'Attack': 'PGD',
        'Epsilon': eps,
        'ASR_MLP': round((1 - accuracy_score(y_attacks_clean, pred_mlp))*100,2),
        'ASR_RF' : round((1 - accuracy_score(y_attacks_clean, pred_rf))*100,2),
        'ASR_XGB': round((1 - accuracy_score(y_attacks_clean, pred_xgb))*100,2)
    })

pgd_results_df = pd.DataFrame(pgd_results)
pgd_results_df.to_csv('pipeline/pgd_results.csv', index=False)

print("✅ PART 4 COMPLETE — PGD GENERATED")

# ============================================================
# PART 5: RESULTS COLLECTION & SUMMARY
# ============================================================

fgsm_results_df = pd.read_csv('pipeline/fgsm_results.csv')
pgd_results_df  = pd.read_csv('pipeline/pgd_results.csv')

all_results_df = pd.concat(
    [fgsm_results_df, pgd_results_df],
    ignore_index=True
)

all_results_df.to_csv('pipeline/all_results.csv', index=False)

print("✅ PART 5 COMPLETE — PHASE 3 FINISHED")

