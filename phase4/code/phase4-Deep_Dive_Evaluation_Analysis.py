"""
Phase 4: Robustness Evaluation & Adversarial Impact Analysis

Goal:
- Quantify the degradation of intrusion detection performance under
  adversarial conditions.
- Analyze which network features and attack categories are most heavily
  exploited to achieve successful evasion.

Outputs:
- Detailed performance impact statistics including Attack Success Rate (ASR),
  accuracy degradation, and false positive rate (FPR) escalation.
- Per-category vulnerability analysis across DoS, Probe, R2L, and U2R attacks.
- Feature perturbation analysis identifying the most influential network
  attributes targeted by adversarial attacks.
- Quantitative robustness ranking across all evaluated models.
"""


# PART 0: IMPORTS & SETUP
# ============================================================

!pip install adversarial-robustness-toolbox
import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import torch
import torch.nn as nn

from art.estimators.classification import (
    SklearnClassifier, PyTorchClassifier, XGBoostClassifier
)
from art.attacks.evasion import (
    FastGradientMethod, ProjectedGradientDescent
)

# PART 1: PATHS & OUTPUT DIRECTORY
# ============================================================
PIPELINE_PATH = '/kaggle/input/datasets/hanaaelganzory/pipeline-backup-correct'
OUTPUT_PATH   = '/kaggle/working/phase4_outputs'
os.makedirs(OUTPUT_PATH, exist_ok=True)

RANDOM_SEED    = 42
EPSILON_VALUES = [0.01, 0.05, 0.1, 0.2]
PGD_ITERATIONS = 10

print("=" * 60)
print("  PHASE 4 — DEEP DIVE EVALUATION")
print("=" * 60)
print(f"  Pipeline path : {PIPELINE_PATH}")
print(f"  Output path   : {OUTPUT_PATH}")

# PART 2: LOAD ALL ARTIFACTS FROM PHASE 1, 2, 3
# ============================================================

# --- 2.1 Feature Arrays ---
X_train_scaled = np.load(f'{PIPELINE_PATH}/X_train_scaled.npy')
X_test_scaled  = np.load(f'{PIPELINE_PATH}/X_test_scaled.npy')
y_train        = np.load(f'{PIPELINE_PATH}/y_train.npy')
y_test         = np.load(f'{PIPELINE_PATH}/y_test.npy')

# --- 2.2 Pipeline Artifacts ---
scaler               = joblib.load(f'{PIPELINE_PATH}/minmax_scaler.pkl')
encoder              = joblib.load(f'{PIPELINE_PATH}/onehot_encoder.pkl')
numerical_mask_bool  = joblib.load(f'{PIPELINE_PATH}/numerical_mask_bool.pkl')
feature_names_final  = joblib.load(f'{PIPELINE_PATH}/feature_names_final.pkl')
test_attack_cats     = joblib.load(f'{PIPELINE_PATH}/test_attack_categories.pkl')
# --- 2.3 Adversarial Examples from Phase 3 (ε=0.2 only) ---
X_adv_fgsm_02 = np.load(f'{PIPELINE_PATH}/X_adv_fgsm_02.npy')
X_adv_pgd_02  = np.load(f'{PIPELINE_PATH}/X_adv_pgd_02.npy')

# --- 2.4 Baseline Predictions from Phase 2 ---
y_pred_rf  = np.load(f'{PIPELINE_PATH}/y_pred_rf.npy')
y_pred_xgb = np.load(f'{PIPELINE_PATH}/y_pred_xgb.npy')
y_pred_mlp = np.load(f'{PIPELINE_PATH}/y_pred_mlp.npy')

print("\n✅ All artifacts loaded successfully")
print(f"   X_test_scaled shape  : {X_test_scaled.shape}")
print(f"   X_adv_fgsm_02 shape  : {X_adv_fgsm_02.shape}")
print(f"   X_adv_pgd_02  shape  : {X_adv_pgd_02.shape}")
print(f"   test_attack_cats     : {test_attack_cats.nunique()} unique categories")

#Part 3
# ============================================================

# --- 3.1 Load sklearn RF & MLP ---
rf_model  = joblib.load(f'{PIPELINE_PATH}/rf_model.pkl')
mlp_model = joblib.load(f'{PIPELINE_PATH}/mlp_model.pkl')
# --- 3.2 Load XGBoost (native loader) ---
xgb_model = XGBClassifier()
xgb_model.load_model(f'{PIPELINE_PATH}/xgb_model.json')
# --- 3.3 Rebuild PyTorch MLP (identical to Phase 3) ---
class MLPNet(nn.Module):
    def __init__(self, input_dim=122, h1=128, h2=64, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),       nn.ReLU(),
            nn.Linear(h2, num_classes)
        )
    def forward(self, x):
        return self.network(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pt_mlp = MLPNet(input_dim=X_train_scaled.shape[1]).to(device)
pt_mlp.load_state_dict(
    torch.load(f'{PIPELINE_PATH}/pt_mlp.pth', map_location=device)
)
pt_mlp.eval()
# --- 3.4 Wrap with ART ---
art_rf  = SklearnClassifier(model=rf_model,  clip_values=(0, 1))
art_xgb = XGBoostClassifier(model=xgb_model, clip_values=(0, 1),
                              nb_features=122, nb_classes=2)

criterion_pt = nn.CrossEntropyLoss().to(device)
optimizer_pt = torch.optim.Adam(pt_mlp.parameters(), lr=0.001)
art_pt_mlp   = PyTorchClassifier(
    model=pt_mlp, loss=criterion_pt, optimizer=optimizer_pt,
    input_shape=(122,), nb_classes=2, clip_values=(0, 1),
    device_type='gpu' if torch.cuda.is_available() else 'cpu'
)

print(f"\n✅ All models loaded & wrapped")
print(f"   Device: {device}")

# PART 4: RECONSTRUCT correct_attack_idx (CRITICAL)
# Must match EXACTLY the indices used in Phase 3
# Logic: samples that PyTorch MLP correctly classified as Attack
# ============================================================
X_test_float32 = X_test_scaled.astype(np.float32)
y_pred_clean_pt = np.argmax(art_pt_mlp.predict(X_test_float32), axis=1)

correct_attack_idx = np.where(
    (y_test == 1) & (y_pred_clean_pt == 1)
)[0]

X_attacks_clean  = X_test_float32[correct_attack_idx]
y_attacks_clean  = y_test[correct_attack_idx]
cats_attacked    = test_attack_cats.iloc[correct_attack_idx].reset_index(drop=True)

print(f"\n✅ correct_attack_idx reconstructed")
print(f"   Total attacks in test     : {(y_test==1).sum()}")
print(f"   Correctly detected (MLP)  : {len(correct_attack_idx)}")
print(f"   Shape of X_attacks_clean  : {X_attacks_clean.shape}")

# PART 5: CATEGORY MAPPING (same as Phase 2 & 3)
# Maps individual attack names → DoS / Probe / R2L / U2R
# ============================================================
CATEGORY_MAP = {
    'neptune':'DoS','smurf':'DoS','back':'DoS','teardrop':'DoS',
    'pod':'DoS','land':'DoS','apache2':'DoS','udpstorm':'DoS',
    'processtable':'DoS','mailbomb':'DoS','worm':'DoS',
    'satan':'Probe','ipsweep':'Probe','nmap':'Probe',
    'portsweep':'Probe','mscan':'Probe','saint':'Probe',
    'warezclient':'R2L','warezmaster':'R2L','guess_passwd':'R2L',
    'ftp_write':'R2L','imap':'R2L','phf':'R2L','multihop':'R2L',
    'named':'R2L','sendmail':'R2L','snmpguess':'R2L',
    'snmpgetattack':'R2L','httptunnel':'R2L','xlock':'R2L','xsnoop':'R2L',
    'buffer_overflow':'U2R','loadmodule':'U2R','rootkit':'U2R',
    'perl':'U2R','sqlattack':'U2R','xterm':'U2R','ps':'U2R'
}
ATTACK_CATEGORIES = ['DoS', 'Probe', 'R2L', 'U2R']

cats_mapped = cats_attacked.map(CATEGORY_MAP).fillna('Other')

print(f"\n✅ Category mapping applied")
print(f"   Distribution:\n{cats_mapped.value_counts().to_string()}")

# Goal: Compute ASR per attack category, per model,
#       per attack type, across ALL epsilon values
# ============================================================
# ============================================================

print("\n" + "=" * 60)
print("  STEP 4.1 — PER-CLASS VULNERABILITY ANALYSIS")
print("=" * 60)

# --- 5.1.1 Re-generate adversarial examples for ALL epsilons ---
# We need per-epsilon adversarial examples for the full curve
# X_adv_fgsm_02 and X_adv_pgd_02 are only ε=0.2

fgsm_adv_all = {}   # fgsm_adv_all[eps] = X_adv array
pgd_adv_all  = {}   # pgd_adv_all[eps]  = X_adv array

for eps in EPSILON_VALUES:
    print(f"\n  Generating FGSM adversarial examples | ε={eps} ...")
    fgsm = FastGradientMethod(
        estimator=art_pt_mlp, eps=eps, eps_step=eps,
        targeted=False, num_random_init=0, minimal=False
    )
    fgsm_adv_all[eps] = fgsm.generate(
        x=X_attacks_clean, mask=numerical_mask_bool
    )
    print(f"  ✅ FGSM ε={eps} done — shape: {fgsm_adv_all[eps].shape}")
    print(f"  Generating PGD adversarial examples  | ε={eps} ...")
    pgd = ProjectedGradientDescent(
        estimator=art_pt_mlp, eps=eps,
        eps_step=eps/PGD_ITERATIONS,
        max_iter=PGD_ITERATIONS,
        targeted=False, num_random_init=1, verbose=False
    )
    pgd_adv_all[eps] = pgd.generate(
        x=X_attacks_clean, mask=numerical_mask_bool
    )
    print(f"  ✅ PGD  ε={eps} done — shape: {pgd_adv_all[eps].shape}")

# --- 5.1.2 Compute Per-Class ASR for each configuration ---
per_class_records = []

PHASE2_FN = {
    # False Negatives from Phase 2 baseline (pre-attack)
    'RF':  {'DoS': 1617, 'Probe': 646,  'R2L': 2736, 'U2R': 60},
    'XGB': {'DoS': 985,  'Probe': 711,  'R2L': 2691, 'U2R': 53},
    'MLP': {'DoS': 982,  'Probe': 396,  'R2L': 2184, 'U2R': 42},
}

for eps in EPSILON_VALUES:
    for attack_name, adv_dict in [('FGSM', fgsm_adv_all),
                                   ('PGD',  pgd_adv_all)]:
        X_adv = adv_dict[eps]
        # Get predictions from all 3 models
        pred_mlp = np.argmax(art_pt_mlp.predict(X_adv), axis=1)
        pred_rf  = np.argmax(art_rf.predict(X_adv),     axis=1)
        pred_xgb = np.argmax(art_xgb.predict(X_adv),    axis=1)

        for cat in ATTACK_CATEGORIES:
            cat_idx = np.where(cats_mapped == cat)[0]
            if len(cat_idx) == 0:
                continue

            true_cat  = y_attacks_clean[cat_idx]  # all = 1

            asr_mlp = 1 - accuracy_score(true_cat, pred_mlp[cat_idx])
            asr_rf  = 1 - accuracy_score(true_cat, pred_rf[cat_idx])
            asr_xgb = 1 - accuracy_score(true_cat, pred_xgb[cat_idx])
            per_class_records.append({
                'Attack'    : attack_name,
                'Epsilon'   : eps,
                'Category'  : cat,
                'N_Samples' : len(cat_idx),
                'ASR_MLP'   : round(asr_mlp * 100, 2),
                'ASR_RF'    : round(asr_rf  * 100, 2),
                'ASR_XGB'   : round(asr_xgb * 100, 2),
                # PGD advantage over FGSM (filled later)
                'PGD_Advantage_MLP': None,
                'PGD_Advantage_RF' : None,
                'PGD_Advantage_XGB': None,
            })

per_class_df = pd.DataFrame(per_class_records)
        
 # --- 5.1.3 Compute PGD Advantage over FGSM (per category) ---
for cat in ATTACK_CATEGORIES:
    for eps in EPSILON_VALUES:
        fgsm_row = per_class_df[
            (per_class_df['Attack']=='FGSM') &
            (per_class_df['Epsilon']==eps) &
            (per_class_df['Category']==cat)
        ]
        pgd_row  = per_class_df[
            (per_class_df['Attack']=='PGD') &
            (per_class_df['Epsilon']==eps) &
            (per_class_df['Category']==cat)
        ]
        if fgsm_row.empty or pgd_row.empty:
            continue

        fi = fgsm_row.index[0]
        pi = pgd_row.index[0]

        for model in ['MLP', 'RF', 'XGB']:
            adv = pgd_row[f'ASR_{model}'].values[0] - \
                  fgsm_row[f'ASR_{model}'].values[0]
            per_class_df.at[pi, f'PGD_Advantage_{model}'] = round(adv, 2)

        # --- 5.1.4 Critical Threshold Analysis ---
# At which epsilon does each category first exceed 50% ASR?
print("\n  CRITICAL THRESHOLD ANALYSIS (ASR ≥ 50%)")
print("  " + "-" * 55)
print(f"  {'Category':<8} {'Model':<6} {'FGSM Threshold':>16} {'PGD Threshold':>15}")
print("  " + "-" * 55)

threshold_records = []
for cat in ATTACK_CATEGORIES:
    for model in ['MLP', 'RF', 'XGB']:
        for atk in ['FGSM', 'PGD']:
            subset = per_class_df[
                (per_class_df['Attack']   == atk) &
                (per_class_df['Category'] == cat)
            ].sort_values('Epsilon')
            crossed = subset[subset[f'ASR_{model}'] >= 50]
            thresh  = crossed['Epsilon'].min() if not crossed.empty else '>0.2'
            threshold_records.append({
                'Category': cat, 'Model': model,
                'Attack': atk, 'Critical_Threshold': thresh
            })

threshold_df = pd.DataFrame(threshold_records)

# Print summary
for cat in ATTACK_CATEGORIES:
    for model in ['MLP', 'RF', 'XGB']:
        fgsm_t = threshold_df[
            (threshold_df['Category']==cat) &
            (threshold_df['Model']==model) &
            (threshold_df['Attack']=='FGSM')
        ]['Critical_Threshold'].values[0]
        pgd_t  = threshold_df[
            (threshold_df['Category']==cat) &
            (threshold_df['Model']==model) &
            (threshold_df['Attack']=='PGD')
        ]['Critical_Threshold'].values[0]
        print(f"  {cat:<8} {model:<6} {str(fgsm_t):>16} {str(pgd_t):>15}")

# --- 5.1.5 Link to Phase 2 Blind Spots ---
print("\n  BLIND SPOT AMPLIFICATION (Phase 2 FN → Phase 4 ASR@ε=0.2)")
print("  " + "-" * 60)
for model in ['RF', 'XGB', 'MLP']:
    print(f"\n  [{model}]")
    for cat in ATTACK_CATEGORIES:
        fn_count = PHASE2_FN[model].get(cat, 0)
        row = per_class_df[
            (per_class_df['Attack']   == 'PGD') &
            (per_class_df['Epsilon']  == 0.2)   &
            (per_class_df['Category'] == cat)
        ]
        if row.empty:
            continue
        asr_val = row[f'ASR_{model}'].values[0]
        print(f"    {cat:<6} → Phase 2 FN: {fn_count:>5} | "
              f"PGD ASR@0.2: {asr_val:.2f}%")
        
# Goal: Understand WHAT the attacker changed and HOW MUCH
#       Using both normalized [0,1] and original units
# ============================================================
# ============================================================

print("\n" + "=" * 60)
print("  STEP 4.2 — FEATURE PERTURBATION PROFILING")
print("=" * 60)

# Numerical feature names only (indices 0–37)
NUM_FEATURE_NAMES = [f for f, m in
                     zip(feature_names_final, numerical_mask_bool) if m]

# --- 5.2.1 Overall Delta (FGSM vs PGD at ε=0.2) ---
def compute_delta(X_orig, X_adv, n_num=38):
    """Compute normalized delta for numerical features only."""
    delta_abs  = np.abs(X_adv[:, :n_num] - X_orig[:, :n_num])
    delta_sign = (X_adv[:, :n_num] - X_orig[:, :n_num])
    mean_abs   = delta_abs.mean(axis=0)
    mean_sign  = delta_sign.mean(axis=0)
    return mean_abs, mean_sign

delta_fgsm_abs, delta_fgsm_sign = compute_delta(
    X_attacks_clean, fgsm_adv_all[0.2])
delta_pgd_abs,  delta_pgd_sign  = compute_delta(
    X_attacks_clean, pgd_adv_all[0.2])

# --- 5.2.2 Inverse Transform → Physical Units ---
# We reconstruct full-size array for inverse_transform
# (scaler was fitted on numerical features only)
def to_physical_delta(X_orig_num, X_adv_num, scaler):
    """
    Convert scaled perturbation to original units.
    X_orig_num, X_adv_num: arrays of shape (n, 38)
    Returns delta in original units.
    """
    orig_inv = scaler.inverse_transform(X_orig_num)
    adv_inv  = scaler.inverse_transform(X_adv_num)
    return adv_inv - orig_inv   # signed physical delta

phys_fgsm = to_physical_delta(
    X_attacks_clean[:, :38], fgsm_adv_all[0.2][:, :38], scaler)
phys_pgd  = to_physical_delta(
    X_attacks_clean[:, :38], pgd_adv_all[0.2][:, :38],  scaler)

mean_phys_fgsm = phys_fgsm.mean(axis=0)
mean_phys_pgd  = phys_pgd.mean(axis=0)

# --- 5.2.3 Build Summary DataFrame (Top 10) ---
feat_df = pd.DataFrame({
    'Feature'           : NUM_FEATURE_NAMES,
    'FGSM_Delta_Norm'   : delta_fgsm_abs,
    'FGSM_Direction'    : np.where(delta_fgsm_sign > 0, 'increase', 'decrease'),
    'FGSM_Physical'     : mean_phys_fgsm,
    'PGD_Delta_Norm'    : delta_pgd_abs,
    'PGD_Direction'     : np.where(delta_pgd_sign  > 0, 'increase', 'decrease'),
    'PGD_Physical'      : mean_phys_pgd,
}).sort_values('PGD_Delta_Norm', ascending=False)

TOP5 = feat_df.head(5)

print("\n  TOP 5 MOST PERTURBED FEATURES (ε=0.2)")
print("  " + "-" * 70)
print(f"  {'Feature':<30} {'FGSM Δ':>8} {'Dir':>9} "
      f"{'PGD Δ':>8} {'Dir':>9}")
print("  " + "-" * 70)
for _, row in TOP5.iterrows():
    print(f"  {row['Feature']:<30} "
          f"{row['FGSM_Delta_Norm']:>8.4f} {row['FGSM_Direction']:>9} "
          f"{row['PGD_Delta_Norm']:>8.4f} {row['PGD_Direction']:>9}")

print("\n  TOP 5 — PHYSICAL UNITS (original scale)")
print("  " + "-" * 60)
for _, row in TOP5.iterrows():
    print(f"  {row['Feature']:<30} "
          f"FGSM: {row['FGSM_Physical']:>+12.2f} | "
          f"PGD:  {row['PGD_Physical']:>+12.2f}")


# --- 5.2.4 Verify Categorical Mask = 0 ---
cat_delta_fgsm = np.abs(
    fgsm_adv_all[0.2][:, 38:] - X_attacks_clean[:, 38:]).max()
cat_delta_pgd  = np.abs(
    pgd_adv_all[0.2][:, 38:]  - X_attacks_clean[:, 38:]).max()
print(f"\n  Categorical Mask Verification:")
print(f"   Max delta (FGSM, cat cols): {cat_delta_fgsm:.6f} "
      f"{'✅ Protected' if cat_delta_fgsm < 1e-6 else '❌ MASK FAILED'}")
print(f"   Max delta (PGD,  cat cols): {cat_delta_pgd:.6f}  "
      f"{'✅ Protected' if cat_delta_pgd  < 1e-6 else '❌ MASK FAILED'}")

# --- 5.2.5 Per-Category Feature Analysis (DoS vs R2L) ---
print("\n  PER-CATEGORY FEATURE PERTURBATION (Top 3, ε=0.2)")
print("  " + "-" * 55)

per_cat_feat_records = []
for cat in ATTACK_CATEGORIES:
    cat_idx = np.where(cats_mapped == cat)[0]
    if len(cat_idx) == 0:
        continue

    d_fgsm = np.abs(
        fgsm_adv_all[0.2][cat_idx, :38] -
        X_attacks_clean[cat_idx, :38]
    ).mean(axis=0)

    d_pgd  = np.abs(
        pgd_adv_all[0.2][cat_idx, :38] -
        X_attacks_clean[cat_idx, :38]
    ).mean(axis=0)

    top3_fgsm = np.argsort(d_fgsm)[::-1][:3]
    top3_pgd  = np.argsort(d_pgd)[::-1][:3]

    print(f"\n  [{cat}] (n={len(cat_idx)})")
    print(f"    FGSM Top3: "
          f"{[NUM_FEATURE_NAMES[i] for i in top3_fgsm]}")
    print(f"    PGD  Top3: "
          f"{[NUM_FEATURE_NAMES[i] for i in top3_pgd]}")

    for i, fi in enumerate(top3_fgsm):
        per_cat_feat_records.append({
            'Category': cat, 'Attack': 'FGSM',
            'Rank': i+1, 'Feature': NUM_FEATURE_NAMES[fi],
            'Delta_Norm': round(d_fgsm[fi], 5)
        })
    for i, fi in enumerate(top3_pgd):
        per_cat_feat_records.append({
            'Category': cat, 'Attack': 'PGD',
            'Rank': i+1, 'Feature': NUM_FEATURE_NAMES[fi],
            'Delta_Norm': round(d_pgd[fi], 5)
        })

per_cat_feat_df = pd.DataFrame(per_cat_feat_records)

# --- 5.2.6 Save ---
feat_df.to_csv(f'{OUTPUT_PATH}/phase4_feature_perturbation.csv', index=False)
per_cat_feat_df.to_csv(
    f'{OUTPUT_PATH}/phase4_per_cat_feature.csv', index=False)

print(f"\n✅ Saved → phase4_feature_perturbation.csv")
print(f"✅ Saved → phase4_per_cat_feature.csv")

# Goal: Rank models using a quantitative robustness score
#       + False Positive analysis after attack
# ============================================================
# ============================================================

print("\n" + "=" * 60)
print("  STEP 4.3 — ROBUSTNESS RANKING SYNTHESIS")
print("=" * 60)

# --- 5.3.1 Baseline Accuracies (from Phase 2) ---
BASELINE = {
    'RF' : 0.7642,
    'XGB': 0.7909,
    'MLP': 0.8062,
}

# --- 5.3.2 ASR at ε=0.2 (PGD, from Phase 3 all_results) ---
ASR_PGD_02 = {
    'RF' : 0.3741,
    'XGB': 0.3391,
    'MLP': 0.9674,
}

# --- 5.3.3 Robustness Score Formula ---
# Robustness Score = Baseline_Accuracy × (1 - ASR_PGD@ε=0.2)
# Intuition: a model that is both accurate AND resistant scores high
ranking_records = []
for model in ['RF', 'XGB', 'MLP']:
    score = BASELINE[model] * (1 - ASR_PGD_02[model])
    ranking_records.append({
        'Model'            : model,
        'Baseline_Acc'     : round(BASELINE[model] * 100, 2),
        'ASR_PGD_0.2'      : round(ASR_PGD_02[model] * 100, 2),
        'Robustness_Score' : round(score, 4),
        'Rank'             : None
    })

ranking_df = pd.DataFrame(ranking_records).sort_values(
    'Robustness_Score', ascending=False).reset_index(drop=True)
ranking_df['Rank'] = ranking_df.index + 1

print("\n  QUANTITATIVE ROBUSTNESS RANKING")
print("  Formula: Score = Baseline_Acc × (1 - ASR_PGD@0.2)")
print("  " + "-" * 55)
print(ranking_df.to_string(index=False))

# --- 5.3.4 Practical Recommendations ---
print("\n  DEPLOYMENT RECOMMENDATIONS")
print("  " + "-" * 55)
recs = {
    'RF' : ('High-Risk',   'Adversarial attacks expected',
            'Maximum robustness (Score=%.4f)' % ranking_df[
                ranking_df['Model']=='RF']['Robustness_Score'].values[0]),
    'XGB': ('Balanced',    'Moderate threat environment',
            'Good accuracy + moderate robustness'),
    'MLP': ('Low-Risk',    'No adversarial attacks expected',
            'Highest accuracy but collapses under attack'),
}
for model, (env, cond, reason) in recs.items():
    print(f"\n  [{model}] — {env} Environment")
    print(f"    Condition : {cond}")
    print(f"    Rationale : {reason}")

# --- 5.3.5 FALSE POSITIVE ANALYSIS ---
# "The Hidden Cost" — Normal traffic misclassified as Attack
# after adversarial perturbation.
# NOTE: We apply the adversarial perturbation that was CRAFTED
# on attack samples, but now we evaluate it on normal samples
# to measure collateral damage.
print("\n  FALSE POSITIVE ANALYSIS AFTER ATTACK")
print("  " + "-" * 55)

# Extract normal samples from test set
normal_idx = np.where(y_test == 0)[0]
X_normal   = X_test_scaled[normal_idx].astype(np.float32)
y_normal   = y_test[normal_idx]

# Baseline FPR (pre-attack)
fp_base_rf  = (np.argmax(art_rf.predict(X_normal),     axis=1) == 1).mean()
fp_base_xgb = (np.argmax(art_xgb.predict(X_normal),    axis=1) == 1).mean()
fp_base_mlp = (np.argmax(art_pt_mlp.predict(X_normal), axis=1) == 1).mean()

# Generate adversarial perturbation on NORMAL samples (ε=0.2)
# This simulates what happens when the IDS receives perturbed
# normal packets — does it start over-alerting?
print("\n  Generating adversarial perturbation on normal samples...")

fgsm_normal = FastGradientMethod(
    estimator=art_pt_mlp, eps=0.2, eps_step=0.2,
    targeted=False, num_random_init=0, minimal=False
)
X_adv_normal_fgsm = fgsm_normal.generate(
    x=X_normal, mask=numerical_mask_bool)

pgd_normal = ProjectedGradientDescent(
    estimator=art_pt_mlp, eps=0.2,
    eps_step=0.2/PGD_ITERATIONS, max_iter=PGD_ITERATIONS,
    targeted=False, num_random_init=1, verbose=False
)
X_adv_normal_pgd = pgd_normal.generate(
    x=X_normal, mask=numerical_mask_bool)

# FPR after FGSM
fp_fgsm_rf  = (np.argmax(art_rf.predict(X_adv_normal_fgsm),     axis=1)==1).mean()
fp_fgsm_xgb = (np.argmax(art_xgb.predict(X_adv_normal_fgsm),   axis=1)==1).mean()
fp_fgsm_mlp = (np.argmax(art_pt_mlp.predict(X_adv_normal_fgsm),axis=1)==1).mean()

# FPR after PGD
fp_pgd_rf   = (np.argmax(art_rf.predict(X_adv_normal_pgd),      axis=1)==1).mean()
fp_pgd_xgb  = (np.argmax(art_xgb.predict(X_adv_normal_pgd),     axis=1)==1).mean()
fp_pgd_mlp  = (np.argmax(art_pt_mlp.predict(X_adv_normal_pgd),  axis=1)==1).mean()

fpr_records = [
    {'Model':'RF',  'FPR_Baseline':round(fp_base_rf*100,2),
     'FPR_FGSM':round(fp_fgsm_rf*100,2),
     'FPR_PGD':round(fp_pgd_rf*100,2),
     'Increase_PGD':round((fp_pgd_rf-fp_base_rf)*100,2)},
    {'Model':'XGB', 'FPR_Baseline':round(fp_base_xgb*100,2),
     'FPR_FGSM':round(fp_fgsm_xgb*100,2),
     'FPR_PGD':round(fp_pgd_xgb*100,2),
     'Increase_PGD':round((fp_pgd_xgb-fp_base_xgb)*100,2)},
    {'Model':'MLP', 'FPR_Baseline':round(fp_base_mlp*100,2),
     'FPR_FGSM':round(fp_fgsm_mlp*100,2),
     'FPR_PGD':round(fp_pgd_mlp*100,2),
     'Increase_PGD':round((fp_pgd_mlp-fp_base_mlp)*100,2)},
]
fpr_df = pd.DataFrame(fpr_records)

print("\n  FALSE POSITIVE RATE — BEFORE & AFTER ATTACK")
print("  " + "-" * 60)
print(f"  {'Model':<6} {'Baseline FPR':>14} {'FGSM FPR':>10} "
      f"{'PGD FPR':>10} {'Δ PGD':>8}")
print("  " + "-" * 60)
for _, row in fpr_df.iterrows():
    print(f"  {row['Model']:<6} {row['FPR_Baseline']:>13.2f}% "
          f"{row['FPR_FGSM']:>9.2f}% "
          f"{row['FPR_PGD']:>9.2f}% "
          f"{row['Increase_PGD']:>+7.2f}%")
# --- 5.3.6 Master Ranking Table ---
# Merge ranking + FPR into one final table
master_ranking = ranking_df.merge(fpr_df, on='Model')
master_ranking.to_csv(f'{OUTPUT_PATH}/phase4_ranking.csv', index=False)
fpr_df.to_csv(f'{OUTPUT_PATH}/phase4_fpr.csv', index=False)

print(f"\n✅ Saved → phase4_ranking.csv")
print(f"✅ Saved → phase4_fpr.csv")

print("\n" + "=" * 60)
print("  PHASE 4 — COMPLETE SUMMARY")
print("=" * 60)

print("\n  📁 Output Files:")
for f in os.listdir(OUTPUT_PATH):
    size = os.path.getsize(f'{OUTPUT_PATH}/{f}')
    print(f"   ✅ {f:<45} ({size} bytes)")

print("\n  🏆 Final Robustness Ranking:")
for _, row in ranking_df.iterrows():
    print(f"   #{int(row['Rank'])} {row['Model']:<4} — "
          f"Score: {row['Robustness_Score']:.4f} | "
          f"Baseline: {row['Baseline_Acc']}% | "
          f"ASR(PGD@0.2): {row['ASR_PGD_0.2']}%")

print("\n  🔍 Key Findings:")
print("   1. MLP most vulnerable (White-box): ASR up to 97.43% (FGSM)")
print("   2. RF most robust to Transfer Attacks (Black-box)")
print("   3. PGD > FGSM on tree-based models at high epsilon")
print("   4. R2L category: highest pre-existing blind spot → "
      "amplified post-attack")
print("   5. False Positives: MLP shows significant FPR increase "
      "under PGD (analyst overload risk)")

print("\n" + "=" * 60)
print("  ✅ PHASE 4 COMPLETE — READY FOR PHASE 5 (VISUALIZATION)")
print("=" * 60)