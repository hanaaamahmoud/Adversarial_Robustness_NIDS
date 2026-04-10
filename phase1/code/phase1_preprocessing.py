"""
Phase 1: Data Preprocessing & Pipeline Construction

Goal:
- Prepare the NSL-KDD dataset for downstream machine learning tasks.
- Encode categorical features and scale numerical features.
- Construct an ART-compatible feature mask that enforces semantic constraints
  (i.e., only numerical features are perturbable under adversarial attacks).

Outputs:
- Scaled training and test feature matrices.
- Binary label arrays for model training and evaluation.
- Pipeline artifacts (encoders, scaler, feature indices, and masks) generated
  locally for reuse in subsequent phases (not committed to the repository).
"""

# ============================================================
# PART 1: IMPORTS & GLOBAL CONFIGURATION
# ============================================================

import numpy as np
import pandas as pd

#Preprocessing
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

#Persistence (Saving the Pipeline)
import joblib

#Utilities
import os
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
EPSILON_VALUES = [0.01, 0.05, 0.1, 0.2]
PGD_ITERATIONS = 10

TRAIN_PATH = 'KDDTrain+.txt'
TEST_PATH = 'KDDTest+.txt

# NSL-KDD COLUMN DEFINITIONS (41 features + label + difficulty)
COLUMN_NAMES = [
'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate',
'label', 'difficulty_level' # will be dropped later
]

# Semantic split (Red Line: Categorical features are NEVER perturbed)
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
NUMERICAL_FEATURES = [
col for col in COLUMN_NAMES
if col not in CATEGORICAL_FEATURES + ['label', 'difficulty_level']
]

print("  Part 1 Complete — Configuration Loaded")
print(f" Numerical features : {len(NUMERICAL_FEATURES)}")
print(f" Categorical features: {len(CATEGORICAL_FEATURES)}")
print(f" Random Seed : {RANDOM_SEED}")
print(f" Epsilon Values : {EPSILON_VALUES}")


# ============================================================
# Part 2: Data Loading & Binary Labeling
# ============================================================


# --- 2.1 Load Raw Data ---
train_df = pd.read_csv(TRAIN_PATH, names=COLUMN_NAMES)
test_df = pd.read_csv(TEST_PATH, names=COLUMN_NAMES)

print(f"  Data Loaded Successfully")
print(f" Train shape: {train_df.shape}")
print(f" Test shape : {test_df.shape}")

# --- 2.2 Save Original Attack Categories (Before Binary Conversion) ---
# IMPORTANT: We preserve the original labels NOW before they are lost.
# These will be used later in Phase 4 for Per-class Impact Analysis
# (DoS, Probe, U2R, R2L vs. Normal)
train_attack_categories = train_df['label'].str.strip('.')
test_attack_categories = test_df['label'].str.strip('.')
print(f"\n  Original Attack Categories Saved")
print(f" Unique categories in Train : {train_attack_categories.nunique()}")
print(f" Unique categories in Test : {test_attack_categories.nunique()}")

# --- 2.3 Binary Labeling --- FIXED
# Convert to string first to handle any NaN values
train_df['binary_label'] = train_df['label'].astype(str).apply(
    lambda x: 0 if 'normal' in x else 1
)
test_df['binary_label'] = test_df['label'].astype(str).apply(
    lambda x: 0 if 'normal' in x else 1
)

print(f"\n✅ Binary Labeling Complete")
print(f"   Train → Normal: {(train_df['binary_label']==0).sum()} | "
      f"Attack: {(train_df['binary_label']==1).sum()}")
print(f"   Test  → Normal: {(test_df['binary_label']==0).sum()} | "
      f"Attack: {(test_df['binary_label']==1).sum()}")

# --- 2.4 Separate Features & Labels ---
# Drop: 'label' (original), 'binary_label' (target), 'difficulty_level' (meta)
X_train_raw = train_df.drop(['label', 'binary_label', 'difficulty_level'], axis=1)
X_test_raw = test_df.drop(['label', 'binary_label', 'difficulty_level'], axis=1)

y_train = train_df['binary_label'].values
y_test = test_df['binary_label'].values

print(f"\n  Features & Labels Separated")
print(f" X_train_raw shape : {X_train_raw.shape} → (samples × 41 features)")
print(f" X_test_raw shape : {X_test_raw.shape} → (samples × 41 features)")
print(f" y_train shape : {y_train.shape}")
print(f" y_test shape : {y_test.shape}")


# ============================================================
# Part 3: One-Hot Encoding
# ============================================================

# --- 3.1 Separate Categorical & Numerical Columns ---
X_train_cat = X_train_raw[CATEGORICAL_FEATURES]
X_test_cat = X_test_raw[CATEGORICAL_FEATURES]

X_train_num = X_train_raw[NUMERICAL_FEATURES]
X_test_num = X_test_raw[NUMERICAL_FEATURES]

print("  Columns Separated")
print(f" Categorical → {CATEGORICAL_FEATURES}")
print(f" Numerical → {len(NUMERICAL_FEATURES)} features")

# --- 3.2 Fit One-Hot Encoder on Train ONLY ---
# handle_unknown='ignore' → if Test has unseen category, output all zeros
# sparse_output=False → return NumPy array (not sparse matrix)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X_train_cat)

# --- 3.3 Transform Both Train & Test ---
X_train_cat_encoded = encoder.transform(X_train_cat)
X_test_cat_encoded = encoder.transform(X_test_cat)

# --- 3.4 Get Encoded Feature Names ---
encoded_cat_feature_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)

print(f"\n  One-Hot Encoding Complete")
print(f" Encoded categorical columns : {len(encoded_cat_feature_names)}")
print(f" Sample encoded names : {list(encoded_cat_feature_names[:5])}")
print(f"\n X_train_cat_encoded shape : {X_train_cat_encoded.shape}")
print(f" X_test_cat_encoded shape : {X_test_cat_encoded.shape}")


# ============================================================
# PART 4: MIN-MAX SCALING (Numerical Features Only)
# ============================================================


# --- 4.1 Fit MinMaxScaler on Train ONLY ---
# Scale all numerical features to [0,1]
# This makes epsilon directly interpretable:
# epsilon=0.1 → max 10% change per feature
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train_num)

# --- 4.2 Transform Both Train & Test ---
X_train_num_scaled = scaler.transform(X_train_num)
X_test_num_scaled  = scaler.transform(X_test_num)

print("✅ Min-Max Scaling Complete")
print(f"   X_train_num_scaled shape : {X_train_num_scaled.shape}")
print(f"   X_test_num_scaled shape  : {X_test_num_scaled.shape}")
print(f"   Train min: {X_train_num_scaled.min():.4f} | "
      f"Train max: {X_train_num_scaled.max():.4f}")
print(f"   Test  min: {X_test_num_scaled.min():.4f}  | "
      f"Test  max: {X_test_num_scaled.max():.4f}")

# --- 4.3 Concatenate Numerical + Categorical ---
# Order: [Numerical (scaled) | Categorical (encoded)]
# This order MUST stay consistent across all phases
X_train_scaled = np.concatenate([X_train_num_scaled,
                                  X_train_cat_encoded], axis=1)
X_test_scaled  = np.concatenate([X_test_num_scaled,
                                  X_test_cat_encoded],  axis=1)

# --- 4.4 Build Final Feature Names List ---
# Numerical names + Encoded categorical names
feature_names_final = (
    list(NUMERICAL_FEATURES) +
    list(encoded_cat_feature_names)
)

print(f"\n✅ Concatenation Complete")
print(f"   X_train_scaled shape : {X_train_scaled.shape}")
print(f"   X_test_scaled shape  : {X_test_scaled.shape}")
print(f"   Total feature names  : {len(feature_names_final)}")


# ============================================================
# PART 5: SAVING PIPELINE & NUMERICAL INDICES
# ============================================================

# --- 5.1 Create Output Directory ---
os.makedirs('pipeline', exist_ok=True)

# --- 5.2 Save Scaler & Encoder ---
joblib.dump(scaler,   'pipeline/minmax_scaler.pkl')
joblib.dump(encoder,  'pipeline/onehot_encoder.pkl')

print("✅ Scaler & Encoder Saved")
print("   pipeline/minmax_scaler.pkl ✅")
print("   pipeline/onehot_encoder.pkl ✅")

# --- 5.3 Save Numerical Indices (Critical for ART Mask) ---
# These indices tell IBM ART exactly which columns to perturb
# Categorical (One-Hot) columns will NEVER be touched
numerical_indices = list(range(len(NUMERICAL_FEATURES)))

# Boolean mask version (required by some ART versions)
numerical_mask_bool = np.array(
    [True]  * len(NUMERICAL_FEATURES) +
    [False] * len(encoded_cat_feature_names)
)


joblib.dump(numerical_indices, 'pipeline/numerical_indices.pkl')
joblib.dump(numerical_mask_bool, 'pipeline/numerical_mask_bool.pkl')

print(f"✅ Numerical Mask Boolean Shape : {numerical_mask_bool.shape}")
print(f"   True  (perturb)  : {numerical_mask_bool.sum()}")
print(f"   False (protected): {(~numerical_mask_bool).sum()}")

# --- 5.4 Save Feature Names (Critical for Phase 4 Analysis) ---
joblib.dump(feature_names_final, 'pipeline/feature_names_final.pkl')

print(f"\n✅ Feature Names Saved")
print(f"   pipeline/feature_names_final.pkl ✅")

# --- 5.5 Save Attack Categories (For Per-class Analysis) ---
joblib.dump(test_attack_categories,  'pipeline/test_attack_categories.pkl')
joblib.dump(train_attack_categories, 'pipeline/train_attack_categories.pkl')

print(f"\n✅ Attack Categories Saved")
print(f"   pipeline/test_attack_categories.pkl  ✅")
print(f"   pipeline/train_attack_categories.pkl ✅")

# --- 5.6 Save Final Arrays ---
np.save('pipeline/X_train_scaled.npy', X_train_scaled)
np.save('pipeline/X_test_scaled.npy',  X_test_scaled)
np.save('pipeline/y_train.npy',        y_train)
np.save('pipeline/y_test.npy',         y_test)

print(f"\n✅ Final Arrays Saved")
print(f"   pipeline/X_train_scaled.npy ✅")
print(f"   pipeline/X_test_scaled.npy  ✅")
print(f"   pipeline/y_train.npy        ✅")
print(f"   pipeline/y_test.npy         ✅")

print("\n" + "="*50)
print("✅ PHASE 1 PIPELINE SAVED SUCCESSFULLY")
print("="*50)

# ============================================================
# PART 6: FINAL VERIFICATION
# ============================================================

# --- 6.1 Array Shapes ---
print("\n📐 Array Shapes:")
print(f"   X_train_scaled : {X_train_scaled.shape}")
print(f"   X_test_scaled  : {X_test_scaled.shape}")
print(f"   y_train        : {y_train.shape}")
print(f"   y_test         : {y_test.shape}")

# --- 6.2 Feature Count Breakdown ---
print("\n Feature Breakdown:")
print(f"   Numerical  (scaled)  : {len(NUMERICAL_FEATURES)}")
print(f"   Categorical (encoded): {len(encoded_cat_feature_names)}")
print(f"   ─────────────────────────────")
print(f"   Total Features       : {X_train_scaled.shape[1]}")

# --- 6.3 Label Distribution ---
print("\n🏷️  Label Distribution:")
print(f"   Train → Normal: {(y_train==0).sum()} | "
      f"Attack: {(y_train==1).sum()}")
print(f"   Test  → Normal: {(y_test==0).sum()}  | "
      f"Attack: {(y_test==1).sum()}")


# --- 6.4 Scaling Sanity Check ---
print("\n🔢 Scaling Sanity Check:")
print(f"   Train → min: {X_train_scaled[:, :38].min():.4f} | "
      f"max: {X_train_scaled[:, :38].max():.4f}  ✅ [0,1]")
print(f"   Test  → min: {X_test_scaled[:, :38].min():.4f}  | "
      f"max: {X_test_scaled[:, :38].max():.4f}  ✅ (Zero-day values expected)")

# --- 6.5 ART Mask Verification ---
print("\n🎯 IBM ART Mask Verification:")
print(f"   Numerical indices  : 0 → {len(NUMERICAL_FEATURES)-1}")
print(f"   Categorical indices: {len(NUMERICAL_FEATURES)} → "
      f"{X_train_scaled.shape[1]-1}")
print(f"   ART will perturb   : columns 0–{len(NUMERICAL_FEATURES)-1} ONLY 🔴")

# --- 6.6 Saved Files Check ---
print("\n💾 Saved Pipeline Files:")
pipeline_files = [
    'minmax_scaler.pkl',
    'onehot_encoder.pkl',
    'numerical_indices.pkl',
    'feature_names_final.pkl',
    'test_attack_categories.pkl',
    'train_attack_categories.pkl',
    'X_train_scaled.npy',
    'X_test_scaled.npy',
    'y_train.npy',
    'y_test.npy'
]
for f in pipeline_files:
    path = f'pipeline/{f}'
    status = "✅" if os.path.exists(path) else "❌"
    print(f"   {status} {path}")

print("\n" + "=" * 55)
print("✅ PHASE 1 PREPROCESSING — COMPLETE")
print("=" * 55)