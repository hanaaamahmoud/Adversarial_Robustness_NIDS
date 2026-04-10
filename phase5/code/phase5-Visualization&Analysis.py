"""
Phase 5: Results Visualization & Adversarial Robustness Insights

Goal:
- Synthesize experimental findings into clear and interpretable visual
  representations.
- Facilitate comparative analysis of model robustness and extract
  actionable security insights.

Outputs:
- Robustness curves illustrating the relationship between perturbation
  magnitude (epsilon) and model accuracy.
- Heatmaps visualizing adversarial vulnerability across different attack
  categories.
- Comparative plots highlighting robustness differences between
  tree-based models and neural networks.
- Final summary figures consolidating key conclusions, including the core
  insight that high classification accuracy does not necessarily imply
  adversarial robustness.
"""

# STEP 1.1: IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import os
import warnings
warnings.filterwarnings('ignore')

# STEP 1.2: GLOBAL STYLE CONFIGURATION
# Cybersecurity dark theme — consistent across ALL plots
# ============================================================

# --- Color Palette ---
COLORS = {
'bg_dark' : '#0D1B2A', # slide/figure background
'bg_panel' : '#1A2940', # panel/axes background
'grid' : '#2A3F5F', # grid lines
'text_white' : '#FFFFFF',
'text_light' : '#B0BEC5',
'FGSM' : '#1E90FF', # blue — FGSM line
'PGD' : '#FF5252', # red — PGD line
'RF' : '#FFB300', # amber — RF model
'XGB' : '#AB47BC', # purple— XGB model
'MLP' : '#00E676', # green — MLP model
'accent_cyan': '#00BCD4',
'danger_red' : '#FF1744',
'safe_green' : '#00C853',
}

# --- Matplotlib Global Theme ---
plt.rcParams.update({
'figure.facecolor' : COLORS['bg_dark'],
'axes.facecolor' : COLORS['bg_panel'],
'axes.edgecolor' : COLORS['grid'],
'axes.labelcolor' : COLORS['text_light'],
'axes.titlecolor' : COLORS['text_white'],
'axes.titlesize' : 14,
'axes.labelsize' : 11,
'xtick.color' : COLORS['text_light'],
'ytick.color' : COLORS['text_light'],
'xtick.labelsize' : 10,
'ytick.labelsize' : 10,
'grid.color' : COLORS['grid'],
'grid.linestyle' : '--',
'grid.alpha' : 0.5,
'legend.facecolor' : COLORS['bg_panel'],
'legend.edgecolor' : COLORS['grid'],
'legend.labelcolor' : COLORS['text_white'],
'legend.fontsize' : 10,
'font.family' : 'DejaVu Sans',
'text.color' : COLORS['text_white'],
'figure.dpi' : 150,
})

# STEP 1.3: GLOBAL CONSTANTS
# Must match Phase 1–4 exactly — never change these
# ============================================================
EPSILON_VALUES = [0.01, 0.05, 0.1, 0.2]
MODELS = ['RF', 'XGB', 'MLP']
CATEGORIES = ['DoS', 'Probe', 'R2L', 'U2R']
ATTACK_TYPES = ['FGSM', 'PGD']

BASELINE_ACC = {
'RF' : 76.42,
'XGB': 79.09,
'MLP': 80.62,
}

# STEP 1.4: OUTPUT DIRECTORY
# All figures saved here — ready for slides & report
# ============================================================
OUTPUT_DIR = 'phase5_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"  Output directory ready: {OUTPUT_DIR}/")

# STEP 1.5: LOAD ALL DATA FILES
# ============================================================

# --- Path Definitions ---
PIPELINE_PATH = '/kaggle/input/datasets/hanaaelganzory/pipeline-backup-correct'
PHASE4_PATH   = '/kaggle/working/phase4_outputs'

# --- Master results (Phase 3 outputs) ---
all_results  = pd.read_csv(f'{PIPELINE_PATH}/all_results.csv')

# --- Phase 4 deep-dive outputs ---
per_class_df = pd.read_csv(f'{PHASE4_PATH}/phase4_per_class_asr.csv')
feat_pert_df = pd.read_csv(f'{PHASE4_PATH}/phase4_feature_perturbation.csv')
ranking_df   = pd.read_csv(f'{PHASE4_PATH}/phase4_ranking.csv')
fpr_df       = pd.read_csv(f'{PHASE4_PATH}/phase4_fpr.csv')

# STEP 1.6: VERIFICATION
# ============================================================
print("=" * 50)
print("  PHASE 5 — PART 1: DATA LOADING VERIFICATION")
print("=" * 50)

print(f"\n📂 Files Loaded:")
print(f"   all_results       : {all_results.shape}   ✅")
print(f"   per_class_df      : {per_class_df.shape}  ✅")
print(f"   feat_pert_df      : {feat_pert_df.shape}  ✅")
print(f"   ranking_df        : {ranking_df.shape}    ✅")
print(f"   fpr_df            : {fpr_df.shape}        ✅")

print(f"\n📊 Robustness Ranking:")
for _, row in ranking_df.iterrows():
    print(f"   #{int(row['Rank'])} {row['Model']:<4} "
          f"→ Score: {row['Robustness_Score']:.4f} "
          f"| Baseline: {row['Baseline_Acc']}% "
          f"| ASR(PGD@0.2): {row['ASR_PGD_0.2']}%")

print(f"\n🎯 Attack Categories in per_class_df:")
print(f"   {per_class_df['Category'].unique().tolist()}")

print(f"\n📈 all_results preview:")
print(all_results.to_string(index=False))

print(f"\n✅ Color Palette : Cybersecurity Dark Theme")
print(f"✅ Output Dir    : {OUTPUT_DIR}/")
print(f"✅ DPI           : {plt.rcParams['figure.dpi']}")

print("\n" + "=" * 50)
print("✅ PART 1 — SETUP COMPLETE. READY FOR PART 2.")
print("=" * 50)

import pandas as pd

all_results = pd.read_csv('/kaggle/input/datasets/hanaaelganzory/pipeline-backup-correct/all_results.csv')
print(f"✅ all_results.csv، shape: {all_results.shape}")

# ============================================================
# OVERRIDE: Switch ALL figures to White / Light Theme
# Run this BEFORE Parts 2, 3, 4, 5, 6 — then re-run each part
# ============================================================

COLORS = {
    'bg_dark'     : '#FFFFFF',
    'bg_panel'    : '#F5F7FA',
    'grid'        : '#D0D7E3',
    'text_white'  : '#1A1A2E',
    'text_light'  : '#4A5568',

    # Attack colors
    'FGSM'        : '#1565C0',
    'PGD'         : '#C62828',

    # Model colors
    'RF'          : '#E65100',
    'XGB'         : '#6A1B9A',
    'MLP'         : '#1B5E20',

    # Accents
    'accent_cyan' : '#00838F',
    'danger_red'  : '#B71C1C',
    'safe_green'  : '#2E7D32',
}

# Global matplotlib configuration
plt.rcParams.update({
    'figure.facecolor'   : COLORS['bg_dark'],
    'axes.facecolor'     : COLORS['bg_panel'],
    'axes.edgecolor'     : COLORS['grid'],

    'axes.labelcolor'    : COLORS['text_light'],
    'axes.titlecolor'    : COLORS['text_white'],
    'axes.titlesize'     : 14,
    'axes.labelsize'     : 11,

    'xtick.color'        : COLORS['text_light'],
    'ytick.color'        : COLORS['text_light'],
    'xtick.labelsize'    : 10,
    'ytick.labelsize'    : 10,

    'grid.color'         : COLORS['grid'],
    'grid.linestyle'     : '--',
    'grid.alpha'         : 0.5,

    'legend.facecolor'   : COLORS['bg_panel'],
    'legend.edgecolor'   : COLORS['grid'],
    'legend.labelcolor'  : COLORS['text_white'],
    'legend.fontsize'    : 10,

    'font.family'        : 'DejaVu Sans',
    'text.color'         : COLORS['text_white'],
    'figure.dpi'         : 150,
})

print("✅ Light Theme Applied Successfully\n")
print("📋 Now re-run in order:")
print("   ▶ Part 2 — ASR Curves")
print("   ▶ Part 3 — Heatmaps")
print("   ▶ Part 4 — Feature Profiling")
print("   ▶ Part 5 — Radar Chart")
print("   ▶ Part 6 — FPR & Scatter")
print("\n✅ All figures will be saved with a white background")

# STEP 2.1: PREPARE DATA
# ============================================================
fgsm_data = all_results[all_results['Attack'] == 'FGSM'].sort_values('Epsilon')
pgd_data = all_results[all_results['Attack'] == 'PGD'].sort_values('Epsilon')

asr_data = {
'RF' : {'FGSM': fgsm_data['ASR_RF'].values, 'PGD': pgd_data['ASR_RF'].values},
'XGB': {'FGSM': fgsm_data['ASR_XGB'].values, 'PGD': pgd_data['ASR_XGB'].values},
'MLP': {'FGSM': fgsm_data['ASR_MLP'].values, 'PGD': pgd_data['ASR_MLP'].values},
}

# STEP 2.2: AUTO CAPTIONS
# ============================================================
CAPTIONS = {
'RF': (
"Fig. 1. ASR vs. Perturbation Budget (ε) for Random Forest under FGSM and PGD. "
"RF demonstrates high resilience as a black-box transfer target, with PGD reaching "
"a maximum ASR of 37.41% at ε=0.2, confirming structural robustness against "
"gradient-based evasion."
),
'XGB': (
"Fig. 2. ASR vs. Perturbation Budget (ε) for XGBoost under FGSM and PGD. "
"XGBoost exhibits moderate vulnerability to transfer attacks, with PGD surpassing "
"FGSM at ε≥0.1 (33.85% vs. 17.33%), indicating iterative attacks are more "
"effective against boosting-based architectures at higher perturbation budgets."
),
'MLP': (
"Fig. 3. ASR vs. Perturbation Budget (ε) for MLP under white-box FGSM and PGD. "
"MLP collapses rapidly under both attacks, with ASR exceeding 96% at ε=0.2, "
"demonstrating that gradient accessibility renders neural networks critically "
"vulnerable to adversarial evasion in NIDS deployments."
),
}

model_titles = {
'RF' : 'Random Forest — Black-box Transfer',
'XGB': 'XGBoost — Black-box Transfer',
'MLP': 'MLP — White-box Attack',
}

model_colors = {
'RF' : COLORS['RF'],
'XGB': COLORS['XGB'],
'MLP': COLORS['MLP'],
}

# STEP 2.3: INDIVIDUAL FIGURES — ONE PER MODEL
# ============================================================
for model in MODELS:

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    ax.set_facecolor(COLORS['bg_panel'])

    y_fgsm = asr_data[model]['FGSM']
    y_pgd  = asr_data[model]['PGD']

    # FGSM line
    ax.plot(
        EPSILON_VALUES, y_fgsm,
        color=COLORS['FGSM'], linewidth=2.5,
        marker='o', markersize=8,
        markeredgecolor=COLORS['bg_dark'],
        markeredgewidth=1.5,
        label='FGSM (single-step)',
        linestyle='--',
        zorder=3
    )

    # PGD line
    ax.plot(
        EPSILON_VALUES, y_pgd,
        color=COLORS['PGD'], linewidth=2.5,
        marker='s', markersize=8,
        markeredgecolor=COLORS['bg_dark'],
        markeredgewidth=1.5,
        label='PGD (iterative, 10 steps)',
        linestyle='-',
        zorder=3
    )

    # Shaded area between curves
    ax.fill_between(
        EPSILON_VALUES, y_fgsm, y_pgd,
        alpha=0.12,
        color=model_colors[model]
    )

     # 50% danger threshold
    ax.axhline(
        y=50,
        color=COLORS['danger_red'],
        linewidth=1.2,
        linestyle=':',
        alpha=0.7,
        label='50% ASR Threshold'
    )

    # Value labels on points
    for eps, vf, vp in zip(EPSILON_VALUES, y_fgsm, y_pgd):
        ax.annotate(
            f'{vf:.1f}%',
            xy=(eps, vf),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            fontsize=8.5,
            color=COLORS['FGSM'],
            fontweight='bold'
        )
        ax.annotate(
            f'{vp:.1f}%',
            xy=(eps, vp),
            xytext=(0, -16),
            textcoords='offset points',
            ha='center',
            fontsize=8.5,
            color=COLORS['PGD'],
            fontweight='bold'
        )

        # Axes formatting
    ax.set_xlim(-0.01, 0.22)
    ax.set_ylim(-5, 110)
    ax.set_xticks(EPSILON_VALUES)
    ax.set_xticklabels([f'ε={e}' for e in EPSILON_VALUES])
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))

    ax.set_xlabel('Perturbation Budget (ε)', fontsize=12, labelpad=8)
    ax.set_ylabel('Attack Success Rate (ASR %)', fontsize=12, labelpad=8)
    ax.set_title(
        f'Robustness Curve — {model_titles[model]}',
        fontsize=14,
        fontweight='bold',
        pad=14
    )

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.8)

    # Baseline badge
    ax.text(
        0.98, 0.97,
        f'Baseline Acc: {BASELINE_ACC[model]}%',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=9,
        color=model_colors[model],
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor=COLORS['bg_dark'],
            edgecolor=model_colors[model],
            alpha=0.8
        )
    )

 plt.tight_layout(pad=1.5)
    fname = f'{OUTPUT_DIR}/P5_Part2_ASR_Curve_{model}.png'
    plt.savefig(
        fname,
        dpi=150,
        bbox_inches='tight',
        facecolor=COLORS['bg_dark']
    )
    plt.show()

    print(f"  Saved → {fname}")
    print(f"\n  Caption:\n {CAPTIONS[model]}\n")
    print("-" * 60)

# STEP 2.4: COMBINED FIGURE — ALL 3 MODELS SIDE BY SIDE
# Clean white version with improved annotations
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.patch.set_facecolor('white')

for ax, model in zip(axes, MODELS):

    ax.set_facecolor('white')

    y_fgsm = asr_data[model]['FGSM']
    y_pgd  = asr_data[model]['PGD']

    # FGSM curve
    ax.plot(
        EPSILON_VALUES, y_fgsm,
        color=COLORS['FGSM'],
        linewidth=2.2,
        marker='o',
        markersize=6,
        markeredgecolor='black',
        label='FGSM',
        linestyle='--',
        zorder=3
    )

    # PGD curve
    ax.plot(
        EPSILON_VALUES, y_pgd,
        color=COLORS['PGD'],
        linewidth=2.2,
        marker='s',
        markersize=6,
        markeredgecolor='black',
        label='PGD',
        linestyle='-',
        zorder=3
    )

    # Shaded difference
    ax.fill_between(
        EPSILON_VALUES, y_fgsm, y_pgd,
        alpha=0.12,
        color=model_colors[model]
    )

     # 50% threshold
    ax.axhline(
        y=50,
        color='red',
        linewidth=1.1,
        linestyle=':',
        alpha=0.6
    )

    # ===== Improved value labels (NO overlap) =====
    for eps, vf, vp in zip(EPSILON_VALUES, y_fgsm, y_pgd):

        # FGSM label (higher offset)
        ax.annotate(
            f'{vf:.0f}%',
            xy=(eps, vf),
            xytext=(0, 12),
            textcoords='offset points',
            ha='center',
            fontsize=7.5,
            color=COLORS['FGSM'],
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.7,
                pad=0.8
            )
        )

        # PGD label (lower offset)
        ax.annotate(
            f'{vp:.0f}%',
            xy=(eps, vp),
            xytext=(0, -18),
            textcoords='offset points',
            ha='center',
            fontsize=7.5,
            color=COLORS['PGD'],
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.7,
                pad=0.8
            )
        )


        # PGD label (lower offset)
        ax.annotate(
            f'{vp:.0f}%',
            xy=(eps, vp),
            xytext=(0, -18),
            textcoords='offset points',
            ha='center',
            fontsize=7.5,
            color=COLORS['PGD'],
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.7,
                pad=0.8
            )
        )

    # Axes formatting
    ax.set_xlim(-0.01, 0.22)
    ax.set_ylim(-5, 115)
    ax.set_xticks(EPSILON_VALUES)
    ax.set_xticklabels([f'ε={e}' for e in EPSILON_VALUES], fontsize=9)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))

    ax.set_xlabel('Perturbation Budget (ε)', fontsize=10)

    ax.set_title(
        model_titles[model],
        fontsize=11,
        fontweight='bold',
        color=model_colors[model]
    )

    ax.grid(True, linestyle='--', alpha=0.25)
    ax.legend(fontsize=9, frameon=False)

      # Baseline badge (clean white)
    ax.text(
        0.97, 0.97,
        f'Baseline: {BASELINE_ACC[model]}%',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=8.5,
        color=model_colors[model],
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor=model_colors[model]
        )
    )

# Shared Y label
axes[0].set_ylabel('Attack Success Rate (ASR %)', fontsize=11)

# Figure title
fig.suptitle(
    'Phase 5 — Robustness Curves: ASR vs. Perturbation Budget (ε)',
    fontsize=14,
    fontweight='bold',
    y=1.02
)

plt.tight_layout(pad=1.5)

fname_combined = f'{OUTPUT_DIR}/P5_Part2_ASR_Curves_Combined.png'
plt.savefig(
    fname_combined,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.show()

print(f"Combined figure saved → {fname_combined}")

# STEP 2.5: SUMMARY
# ============================================================
print("\n" + "=" * 60)
print(" PHASE 5 — PART 2: ASR CURVES COMPLETE")
print("=" * 60)
print(f"\n  Saved in: {OUTPUT_DIR}/")
print(f"   P5_Part2_ASR_Curve_RF.png")
print(f"   P5_Part2_ASR_Curve_XGB.png")
print(f"   P5_Part2_ASR_Curve_MLP.png")
print(f"   P5_Part2_ASR_Curves_Combined.png")
print(f"\n  Key Observations:")
print(f" MLP → Collapses rapidly (ASR ~97% at ε=0.2) [White-box]")
print(f" XGB → PGD > FGSM at ε≥0.1 (transfer gap widens)")
print(f" RF → Most robust transfer target (max ASR 37.41%)")
print("\n" + "=" * 60)
print("  PART 2 COMPLETE — READY FOR PART 3 (Heatmap).")
print("=" * 60)

# Two Heatmaps:
# Heatmap A: ASR @ ε=0.2 per (Model × Category) — FGSM vs PGD
# Heatmap B: Critical Threshold (at which ε does ASR hit 50%)
# ============================================================

# ============================================================
# STEP 3.1: PREPARE DATA — ASR @ ε=0.2
# ============================================================

# Filter ε = 0.2 only
per_class_02 = per_class_df[per_class_df['Epsilon'] == 0.2].copy()

# Build matrices: rows = Categories, cols = Models
# Separate FGSM and PGD
def build_asr_matrix(attack_type):

    subset = per_class_02[per_class_02['Attack'] == attack_type]

    matrix = pd.DataFrame(
        index=CATEGORIES,
        columns=MODELS,
        dtype=float
    )

    for _, row in subset.iterrows():
        matrix.loc[row['Category'], 'RF']  = row['ASR_RF']
        matrix.loc[row['Category'], 'XGB'] = row['ASR_XGB']
        matrix.loc[row['Category'], 'MLP'] = row['ASR_MLP']

    return matrix


# ASR matrices at ε = 0.2
fgsm_matrix = build_asr_matrix('FGSM')
pgd_matrix  = build_asr_matrix('PGD')

# ============================================================
# STEP 3.2: PREPARE DATA — CRITICAL THRESHOLD
# At which ε does ASR first exceed 50%?
# ============================================================

def build_threshold_matrix(attack_type):
    """
    Returns:
        matrix       : numeric threshold values (for heatmap coloring)
        label_matrix : string labels (for annotation)
    """

    # Numeric encoding for heatmap colors
    thresh_map = {
        0.01: 1,
        0.05: 2,
        0.10: 3,
        0.20: 4,
        '>0.2': 5
    }

    matrix = pd.DataFrame(index=CATEGORIES, columns=MODELS, dtype=float)
    label_matrix = pd.DataFrame(index=CATEGORIES, columns=MODELS, dtype=str)

    for cat in CATEGORIES:
        for model in MODELS:

            col = f'ASR_{model}'

            subset = (
                per_class_df[
                    (per_class_df['Attack'] == attack_type) &
                    (per_class_df['Category'] == cat)
                ]
                .sort_values('Epsilon')
            )
crossed = subset[subset[col] >= 50]

            if crossed.empty:
                thresh_label = '>0.2'
                thresh_val = 5
            else:
                eps_hit = crossed['Epsilon'].min()
                thresh_label = str(eps_hit)
                thresh_val = thresh_map.get(eps_hit, 5)

            matrix.loc[cat, model] = thresh_val
            label_matrix.loc[cat, model] = thresh_label

    return matrix.astype(float), label_matrix


# Build threshold matrices
fgsm_thresh_num, fgsm_thresh_lbl = build_threshold_matrix('FGSM')
pgd_thresh_num,  pgd_thresh_lbl  = build_threshold_matrix('PGD')


# Sanity checks / inspection
print("  Data prepared")

print(f"\n FGSM ASR Matrix @ ε=0.2:\n{fgsm_matrix}")
print(f"\n PGD ASR Matrix @ ε=0.2:\n{pgd_matrix}")

print(f"\n FGSM Critical Thresholds:\n{fgsm_thresh_lbl}")
print(f"\n PGD Critical Thresholds:\n{pgd_thresh_lbl}")

# ============================================================
# STEP 3.3: CUSTOM COLORMAPS
# Red = high ASR (dangerous), Green = low ASR (safe)
# ============================================================

from matplotlib.colors import LinearSegmentedColormap

# ASR heatmap: green → yellow → red
asr_cmap = LinearSegmentedColormap.from_list(
    'asr_cmap',
    ['#2166AC', '#F7F7F7', '#D6604D'],
    N=256
)

# Threshold heatmap:
# Inverted meaning:
#   Low ε threshold → model collapses early → more dangerous
#   High ε threshold → more robust → safer
# Colors: red (early collapse) → yellow → green (resistant)
thresh_cmap = LinearSegmentedColormap.from_list(
    'thresh_cmap',
    ['#FF1744', '#FFD600', '#00C853'],
    N=256
)

# ============================================================
# STEP 3.4: HEATMAP A — ASR @ ε = 0.2 (FGSM vs PGD)
# Clean Academic Version (White background, Blues colormap)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

heatmap_data = [
    (fgsm_matrix, 'FGSM — ASR @ ε = 0.2', axes[0]),
    (pgd_matrix,  'PGD — ASR @ ε = 0.2', axes[1]),
]

for matrix, title, ax in heatmap_data:

    ax.set_facecolor('white')
    data = matrix.values.astype(float)

    im = ax.imshow(
        data,
        cmap='Blues',
        aspect='auto',
        vmin=0,
        vmax=100
    )

    # Cell annotations
    for i in range(len(CATEGORIES)):
        for j in range(len(MODELS)):
            val = data[i, j]

            ax.text(
                j, i,
                f'{val:.1f}%',
                ha='center',
                va='center',
                fontsize=12,
                color='black'
            )
              # Axes formatting
    ax.set_xticks(range(len(MODELS)))
    ax.set_yticks(range(len(CATEGORIES)))

    ax.set_xticklabels(
        MODELS,
        fontsize=11,
        fontweight='bold',
        rotation=30
    )

    ax.set_yticklabels(
        CATEGORIES,
        fontsize=11,
        fontweight='bold'
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight='bold',
        pad=10
    )

    # Grid-like separation
    ax.set_xticks(np.arange(-.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(CATEGORIES), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('ASR (%)', fontsize=10)

# Figure title
fig.suptitle(
    'Per-Class Vulnerability Heatmap — ASR at ε = 0.2',
    fontsize=14,
    fontweight='bold',
    y=1.03
)

plt.tight_layout(pad=2.0)

fname_a = f'{OUTPUT_DIR}/P5_Part3_Heatmap_ASR.png'
plt.savefig(
    fname_a,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()
print(f"Saved → {fname_a}")

# ============================================================
# STEP 3.5: HEATMAP B — CRITICAL THRESHOLD
# Clean Academic Version (White background, Blues colormap)
# At which ε does each category first exceed 50% ASR?
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

thresh_data = [
    (pgd_thresh_num,  pgd_thresh_lbl,
     'PGD — Critical Threshold (ASR ≥ 50%)', axes[0]),
    (fgsm_thresh_num, fgsm_thresh_lbl,
     'FGSM — Critical Threshold (ASR ≥ 50%)', axes[1]),
]

for num_mat, lbl_mat, title, ax in thresh_data:

    ax.set_facecolor('white')
    data = num_mat.values.astype(float)

    im = ax.imshow(
        data,
        cmap='Blues',
        aspect='auto',
        vmin=1,
        vmax=5
    )

    # Cell annotations
    for i in range(len(CATEGORIES)):
        for j in range(len(MODELS)):
            lbl = lbl_mat.iloc[i, j]
            display = f'ε={lbl}' if lbl != '>0.2' else '>0.2'

            ax.text(
                j, i,
                display,
                ha='center',
                va='center',
                fontsize=11,
                color='black'
            )
  # Axes formatting
    ax.set_xticks(range(len(MODELS)))
    ax.set_yticks(range(len(CATEGORIES)))

    ax.set_xticklabels(
        MODELS,
        fontsize=11,
        fontweight='bold',
        rotation=30
    )

    ax.set_yticklabels(
        CATEGORIES,
        fontsize=11,
        fontweight='bold'
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight='bold',
        pad=10
    )

    # Grid-like separation
    ax.set_xticks(np.arange(-.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(CATEGORIES), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([1, 2, 3, 4, 5])
    cbar.set_ticklabels(
        ['ε=0.01', 'ε=0.05', 'ε=0.1', 'ε=0.2', '>0.2'],
        fontsize=9
    )
    cbar.set_label(
        'First ε where ASR ≥ 50%',
        fontsize=10
    )

    # Figure title (neutral & academic)
fig.suptitle(
    'Critical Threshold Heatmap — First ε Where ASR ≥ 50%',
    fontsize=14,
    fontweight='bold',
    y=1.03
)

plt.tight_layout(pad=2.0)

fname_b = f'{OUTPUT_DIR}/P5_Part3_Heatmap_Threshold.png'
plt.savefig(
    fname_b,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()
print(f"Saved → {fname_b}")

# ============================================================
# STEP 3.6: AUTO CAPTIONS
# ============================================================

CAPTION_HEATMAP_A = (
    "Fig. 4. Per-class vulnerability heatmap showing ASR at ε=0.2 for FGSM (left) "
    "and PGD (right) across all model-category pairs. R2L and U2R categories exhibit "
    "near-total vulnerability (ASR=100%) under both attacks across all models, while "
    "DoS demonstrates the highest structural resistance, particularly in RF and XGB."
)

CAPTION_HEATMAP_B = (
    "Fig. 5. Critical threshold heatmap indicating the minimum perturbation budget (ε) "
    "at which ASR first exceeds 50% per model-category pair. R2L and U2R categories "
    "collapse at ε=0.01 across most models, confirming that rare attack classes represent "
    "the primary exploitable blind spot in ML-based NIDS under adversarial conditions."
)

print(f"\n  Caption A:\n {CAPTION_HEATMAP_A}")
print(f"\n  Caption B:\n {CAPTION_HEATMAP_B}")

# ============================================================
# STEP 3.7: SUMMARY
# ============================================================

print("\n" + "=" * 60)
print(" PHASE 5 — PART 3: HEATMAPS COMPLETE")
print("=" * 60)

print(f"\n  Saved in: {OUTPUT_DIR}/")
print("   P5_Part3_Heatmap_ASR.png")
print("   P5_Part3_Heatmap_Threshold.png")

print("\n  Key Observations:")
print(" R2L → Collapses at ε=0.01 across ALL models (most dangerous)")
print(" U2R → Same as R2L — extreme vulnerability at tiny ε")
print(" DoS → Most resistant category (MLP needs ε=0.1 to reach 50%)")
print(" RF  → Most robust model especially vs DoS and Probe")

print("\n" + "=" * 60)
print("  PART 3 COMPLETE — READY FOR PART 4 (Feature Profiling).")
print("=" * 60)

# ============================================================
# Horizontal Bar Chart — Top 10 most perturbed features
# Shows: Normalized Delta (Δ) + Physical Units
# Side-by-side: FGSM vs PGD at ε = 0.2
# ============================================================

# ============================================================
# STEP 4.1: PREPARE DATA — TOP 10 FEATURES
# Sorted by PGD_Delta_Norm (strongest perturbation)
# ============================================================

# Select top 10 most perturbed features (by PGD)
top10 = feat_pert_df.nlargest(10, 'PGD_Delta_Norm').copy()

# Sort ascending for horizontal bar plotting
top10 = top10.sort_values('PGD_Delta_Norm', ascending=True)

# Extract values
features   = top10['Feature'].tolist()
fgsm_delta = top10['FGSM_Delta_Norm'].tolist()
pgd_delta  = top10['PGD_Delta_Norm'].tolist()

fgsm_phys = top10['FGSM_Physical'].tolist()
pgd_phys  = top10['PGD_Physical'].tolist()

fgsm_dir = top10['FGSM_Direction'].tolist()
pgd_dir  = top10['PGD_Direction'].tolist()

# Clean feature names for display
feat_labels = [f.replace('_', ' ') for f in features]

# Console sanity check
print("  Top 10 features prepared\n")
print(f"{'Feature':<30} {'FGSM Δ':>10} {'PGD Δ':>10}")
print("-" * 54)

for f, fd, pd_ in zip(features, fgsm_delta, pgd_delta):
    print(f"{f:<30} {fd:>10.4f} {pd_:>10.4f}")


    # ============================================================
# STEP 4.2: PLOT A — NORMALIZED DELTA (Clean & Minimal Version)
# Reduced visual clutter — Academic / GitHub style
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

y_pos = np.arange(len(features))
bar_h = 0.35

# PGD bars
ax.barh(
    y_pos + bar_h / 2,
    pgd_delta,
    bar_h,
    color='#1f77b4',
    alpha=0.85,
    label='PGD'
)

# FGSM bars
ax.barh(
    y_pos - bar_h / 2,
    fgsm_delta,
    bar_h,
    color='#aec7e8',
    alpha=0.85,
    label='FGSM'
)

# Value labels (PGD only — reduce clutter)
for i, val in enumerate(pgd_delta):
    ax.text(
        val + 0.003,
        i + bar_h / 2,
        f'{val:.3f}',
        va='center',
        fontsize=8,
        color='black'
    )

# Reference line
ax.axvline(
    x=0.1,
    color='gray',
    linewidth=1.1,
    linestyle=':',
    alpha=0.7
)

# Axes formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(
    feat_labels,
    fontsize=10
)

ax.set_xlabel(
    'Mean Absolute Perturbation (Normalized Δ)',
    fontsize=11
)

ax.set_title(
    'Feature Perturbation Profiling — FGSM vs. PGD (ε = 0.2)',
    fontsize=13,
    fontweight='bold',
    pad=12
)

ax.set_xlim(0, 0.27)
ax.grid(True, axis='x', linestyle='--', alpha=0.25)

ax.legend(
    loc='lower right',
    fontsize=10,
    frameon=False
)

plt.tight_layout(pad=1.5)

fname_norm = f'{OUTPUT_DIR}/P5_Part4_Feature_Normalized.png'
plt.savefig(
    fname_norm,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()
print(f"Saved → {fname_norm}")# Prepare Top-5 features for Phase 5 plots (from Phase 4 outputs)
import pandas as pd
import numpy as np

# Load Phase 4 feature perturbation results
feat_pert_df = pd.read_csv("/kaggle/working/phase4_outputs/phase4_feature_perturbation.csv")

# Select Top-5 features by PGD normalized delta
top5 = feat_pert_df.nlargest(5, "PGD_Delta_Norm").copy()
top5 = top5.sort_values("PGD_Delta_Norm", ascending=True)

feat5   = [f.replace("_", " ") for f in top5["Feature"].tolist()]
fgsm_p5 = top5["FGSM_Physical"].tolist()
pgd_p5  = top5["PGD_Physical"].tolist()

print("feat5 prepared for Phase 5 plots ✅")

# ============================================================
# STEP 4.3: PLOT B — PHYSICAL UNITS (Clean & Minimal Version)
# Top 5 features — Physical perturbation magnitude
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

y5 = np.arange(len(feat5))
bar_h = 0.35

# PGD bars
ax.barh(
    y5 + bar_h / 2,
    pgd_p5,
    bar_h,
    color='#1f77b4',
    alpha=0.85,
    label='PGD'
)

# FGSM bars
ax.barh(
    y5 - bar_h / 2,
    fgsm_p5,
    bar_h,
    color='#aec7e8',
    alpha=0.85,
    label='FGSM'
)

# Value labels (PGD only — reduce clutter)
for i, val in enumerate(pgd_p5):
    if val >= 1:
        label = f'{val:,.0f}'
    else:
        label = f'{val:.3f}'

    ax.text(
        val,
        i + bar_h / 2,
        f'  {label}',
        va='center',
        fontsize=8,
        color='black'
    )
# Define physical units for features
UNITS = {
    'num root': 'count',
    'su attempted': 'binary (0/1)',
    'logged in': 'binary (0/1)',
    'srv diff host rate': 'rate [0–1]',
    'is host login': 'binary (0/1)',
}
# Y-axis labels with physical units (clean)
y_labels_with_units = [
    f'{f}  [{UNITS.get(f, "value")}]' for f in feat5
]

ax.set_yticks(y5)
ax.set_yticklabels(
    y_labels_with_units,
    fontsize=10
)

ax.set_xlabel(
    'Mean Perturbation in Original Units',
    fontsize=11
)

ax.set_title(
    'Top 5 Features — Physical Perturbation Magnitude (ε = 0.2)',
    fontsize=13,
    fontweight='bold',
    pad=12
)

ax.grid(True, axis='x', linestyle='--', alpha=0.25)

ax.legend(
    loc='lower right',
    fontsize=10,
    frameon=False
)

plt.tight_layout(pad=1.5)

fname_phys = f'{OUTPUT_DIR}/P5_Part4_Feature_Physical.png'
plt.savefig(
    fname_phys,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()
print(f"Saved → {fname_phys}")

# ============================================================
# STEP 4.4: AUTO CAPTIONS
# ============================================================

CAPTION_NORM = (
    "Fig. 6. Normalized perturbation magnitude (Δ) for the top 10 most exploited "
    "NSL-KDD features under FGSM and PGD at ε=0.2. The feature num_root receives "
    "the highest perturbation under both attacks (FGSM: Δ=0.200, PGD: Δ=0.196), "
    "followed by su_attempted and logged_in, indicating that privilege-escalation "
    "and session indicators are the primary attack vectors."
)

CAPTION_PHYS = (
    "Fig. 7. Physical-scale perturbation for the top 5 exploited features at ε=0.2. "
    "The num_root feature is incremented by +1,493 (FGSM) and +1,467 (PGD) on average, "
    "while binary features such as su_attempted and logged_in are shifted by +0.34–0.38, "
    "confirming that adversarial perturbations remain semantically plausible within "
    "realistic network traffic constraints."
)

print(f"\n  Caption (Normalized):\n {CAPTION_NORM}")
print(f"\n  Caption (Physical):\n {CAPTION_PHYS}")

# ============================================================
# STEP 4.5: SUMMARY
# ============================================================

print("\n" + "=" * 60)
print(" PHASE 5 — PART 4: FEATURE PROFILING COMPLETE")
print("=" * 60)

print(f"\n  Saved in: {OUTPUT_DIR}/")
print("   P5_Part4_Feature_Normalized.png")
print("   P5_Part4_Feature_Physical.png")

print("\n  Key Observations:")
print(" num_root     → Most exploited (Δ≈0.20, +1,493 count)")
print(" su_attempted → Binary feature shifted by +0.38")
print(" logged_in    → Session indicator exploited in all attacks")
print(" All top features → Direction: INCREASE (privilege escalation)")

print("\n" + "=" * 60)
print("  PART 4 COMPLETE — READY FOR PART 5 (Radar Chart).")
print("=" * 60)

# ============================================================
# STEP 6.1: PREPARE FPR DATA
# ============================================================

models_ordered = ['RF', 'XGB', 'MLP']

# Extract FPR values
fpr_base = [
    fpr_df[fpr_df['Model'] == m]['FPR_Baseline'].values[0]
    for m in models_ordered
]

fpr_fgsm = [
    fpr_df[fpr_df['Model'] == m]['FPR_FGSM'].values[0]
    for m in models_ordered
]

fpr_pgd = [
    fpr_df[fpr_df['Model'] == m]['FPR_PGD'].values[0]
    for m in models_ordered
]

fpr_inc = [
    fpr_df[fpr_df['Model'] == m]['Increase_PGD'].values[0]
    for m in models_ordered
]

# Console sanity check
print("  FPR Data:\n")
print(f"{'Model':<6} {'Baseline':>10} {'FGSM':>10} {'PGD':>10} {'Δ PGD':>10}")
print("-" * 55)

for m, b, f, p, inc in zip(
    models_ordered,
    fpr_base,
    fpr_fgsm,
    fpr_pgd,
    fpr_inc
):
    print(f"{m:<6} {b:>9.2f}% {f:>9.2f}% {p:>9.2f}% {inc:>+9.2f}%")


# ============================================================
# STEP 6.2: PLOT A — FPR COMPARISON (Clean & Minimal Version)
# Reduced visual clutter — Academic / GitHub style
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

x = np.arange(len(models_ordered))
width = 0.3

# Baseline
ax.bar(
    x - width/2,
    fpr_base,
    width,
    color='#aec7e8',
    label='Baseline'
)

# After PGD (worst case — focus on impact)
ax.bar(
    x + width/2,
    fpr_pgd,
    width,
    color='#1f77b4',
    label='After PGD (ε = 0.2)'
)

# Value labels (PGD only)
for i, val in enumerate(fpr_pgd):
    ax.text(
        x[i] + width/2,
        val + 1,
        f'{val:.1f}%',
        ha='center',
        va='bottom',
        fontsize=9
    )# Axes formatting
ax.set_xticks(x)
ax.set_xticklabels(models_ordered, fontsize=11)

ax.set_ylabel('False Positive Rate (FPR %)', fontsize=11)

ax.set_title(
    'False Positive Rate Before and After Adversarial Attack',
    fontsize=13,
    fontweight='bold',
    pad=12
)

ax.set_ylim(0, max(fpr_pgd) * 1.25)

ax.grid(True, axis='y', linestyle='--', alpha=0.25)

ax.legend(
    loc='upper left',
    fontsize=10,
    frameon=False
)

plt.tight_layout()

fname_fpr = f'{OUTPUT_DIR}/P5_Part6_FPR_Bar.png'
plt.savefig(
    fname_fpr,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()
print(f"Saved → {fname_fpr}")

# ============================================================
# STEP 6.3: ACCURACY vs ASR (Clean & Minimal Comparison)
# Replaces scatter plot with simple bar comparison
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Models
models = ['RF', 'XGB', 'MLP']

# Values
acc_vals = [BASELINE_ACC[m] for m in models]
asr_vals = [
    ranking_df[ranking_df['Model'] == m]['ASR_PGD_0.2'].values[0]
    for m in models
]

x = np.arange(len(models))
width = 0.35

# Baseline Accuracy bars
ax.bar(
    x - width / 2,
    acc_vals,
    width,
    color='#aec7e8',
    label='Baseline Accuracy'
)

# ASR bars
ax.bar(
    x + width / 2,
    asr_vals,
    width,
    color='#1f77b4',
    label='ASR (PGD, ε = 0.2)'
)

# Value labels
for i in range(len(models)):
    ax.text(
        x[i] - width / 2,
        acc_vals[i] + 1,
        f'{acc_vals[i]:.1f}%',
        ha='center',
        va='bottom',
        fontsize=9
    )
    ax.text(
        x[i] + width / 2,
        asr_vals[i] + 1,
        f'{asr_vals[i]:.1f}%',
        ha='center',
        va='bottom',
        fontsize=9
    )

# Axes formatting
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('Percentage (%)', fontsize=11)

ax.set_title(
    'Baseline Accuracy vs. Adversarial Vulnerability\n'
    '(PGD Attack at ε = 0.2)',
    fontsize=13,
    fontweight='bold',
    pad=12
)

ax.set_ylim(0, max(max(acc_vals), max(asr_vals)) * 1.25)
ax.grid(True, axis='y', linestyle='--', alpha=0.25)

ax.legend(frameon=False)

plt.tight_layout()

# Save
fname_simple = f'{OUTPUT_DIR}/P5_Part6_AccVsASR.png'
plt.savefig(
    fname_simple,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()
print(f"Saved → {fname_simple}")

# ============================================================
# STEP 6.4: AUTO CAPTIONS
# ============================================================

CAPTION_FPR = (
    "Fig. 9. False positive rate (FPR) before and after adversarial attack at ε=0.2. "
    "All three models exceed the 10% analyst tolerance threshold under both FGSM and PGD, "
    "with MLP reaching 90.81% FPR under PGD (+83.53% increase), rendering it operationally "
    "unusable. RF exhibits the lowest post-attack FPR (50.13%), confirming its superior "
    "stability under adversarial conditions."
)

CAPTION_SCATTER = (
    "Fig. 10. Accuracy–robustness trade-off scatter plot. MLP achieves the highest baseline "
    "accuracy (80.62%) but suffers a PGD ASR of 96.74%, placing it in the high-vulnerability "
    "quadrant. XGBoost achieves the best balance with a robustness score of 0.5227, "
    "while RF maintains the second-best robustness despite a lower baseline accuracy, "
    "demonstrating that accuracy alone is an insufficient security metric."
)

print(f"\n  Caption (FPR Bar):\n {CAPTION_FPR}")
print(f"\n  Caption (Scatter):\n {CAPTION_SCATTER}")


# ============================================================
# STEP 6.5: SUMMARY
# ============================================================

print("\n" + "=" * 60)
print(" PHASE 5 — PART 6: FPR & SCATTER COMPLETE")
print("=" * 60)

print(f"\n  Saved in: {OUTPUT_DIR}/")
print("   P5_Part6_FPR_Bar.png")
print("   P5_Part6_Scatter_AccVsASR.png")

print("\n  Key Observations:")
print(" MLP → FPR jumps from 7.28% → 90.81% under PGD (+83.53%)")
print(" XGB → FPR jumps from 2.84% → 84.41% under FGSM (+81.57%)")
print(" RF  → Most stable FPR: only +47.49% increase")
print(" All → Exceed 10% threshold = analyst alert overload")

print("\n" + "=" * 60)
print("  PART 6 COMPLETE — READY FOR PART 7 (Final Summary).")
print("=" * 60)

# ============================================================
# OVERRIDE: Switch to White / Light Theme for Part 7
# ============================================================

COLORS_LIGHT = {
    'bg_dark'     : '#FFFFFF',
    'bg_panel'    : '#F5F7FA',
    'grid'        : '#D0D7E3',
    'text_white'  : '#1A1A2E',
    'text_light'  : '#4A5568',

    # Attack colors
    'FGSM'        : '#1565C0',   # darker blue
    'PGD'         : '#C62828',   # darker red

    # Model colors
    'RF'          : '#E65100',   # darker amber
    'XGB'         : '#6A1B9A',   # darker purple
    'MLP'         : '#1B5E20',   # darker green

    # Accents
    'accent_cyan' : '#00838F',
    'danger_red'  : '#B71C1C',
    'safe_green'  : '#2E7D32',
}

# Override global COLORS dictionary
COLORS = COLORS_LIGHT

# Update matplotlib defaults for light theme
plt.rcParams.update({
    'figure.facecolor'   : COLORS['bg_dark'],
    'axes.facecolor'     : COLORS['bg_panel'],
    'axes.edgecolor'     : COLORS['grid'],
    'axes.labelcolor'    : COLORS['text_light'],
    'axes.titlecolor'    : COLORS['text_white'],
    'xtick.color'        : COLORS['text_light'],
    'ytick.color'        : COLORS['text_light'],
    'grid.color'         : COLORS['grid'],
    'legend.facecolor'   : COLORS['bg_panel'],
    'legend.edgecolor'   : COLORS['grid'],
    'legend.labelcolor'  : COLORS['text_white'],
    'text.color'         : COLORS['text_white'],
})

print("✅ Theme switched to Light / White — Ready for Part 7")

# ============================================================
# STEP 7.1: FINAL SUMMARY DASHBOARD (Clean & Minimal Version)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ------------------------
# COLOR PALETTE (FIXED)
# ------------------------
BASE_COLOR   = '#aec7e8'   # light blue (baseline / benign)
ATTACK_COLOR = '#1f77b4'   # dark blue (attack / worst-case)
ACCENT_COLOR = '#4c72b0'   # titles / emphasis

# ------------------------
# FIGURE & LAYOUT
# ------------------------
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('white')

gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ============================================================
# Panel 1: Baseline Accuracy
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('white')

bars = ax1.bar(
    MODELS,
    [BASELINE_ACC[m] for m in MODELS],
    color=BASE_COLOR
)

for bar, m in zip(bars, MODELS):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f'{BASELINE_ACC[m]}%',
        ha='center',
        fontsize=9
    )

ax1.set_ylim(70, 85)
ax1.set_ylabel('Accuracy (%)', fontsize=10)
ax1.set_title('① Baseline Accuracy', fontsize=11,
              fontweight='bold', color=ACCENT_COLOR)
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)


# ============================================================
# Panel 2: ASR @ ε = 0.2
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('white')

x2 = np.arange(len(MODELS))
w2 = 0.35

fgsm_asr02 = [fgsm_data[f'ASR_{m}'].values[-1] for m in MODELS]
pgd_asr02  = [pgd_data[f'ASR_{m}'].values[-1] for m in MODELS]

ax2.bar(x2 - w2/2, fgsm_asr02, w2,
        color=BASE_COLOR, label='FGSM')

ax2.bar(x2 + w2/2, pgd_asr02, w2,
        color=ATTACK_COLOR, label='PGD')

ax2.set_xticks(x2)
ax2.set_xticklabels(MODELS, fontsize=10)
ax2.set_ylim(0, 110)
ax2.set_ylabel('ASR (%)', fontsize=10)
ax2.set_title('② ASR @ ε = 0.2', fontsize=11,
              fontweight='bold', color=ACCENT_COLOR)
ax2.legend(frameon=False, fontsize=9)
ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

# ============================================================
# Panel 3: Robustness Score
# ============================================================
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('white')

scores = [
    ranking_df[ranking_df['Model'] == m]['Robustness_Score'].values[0]
    for m in MODELS
]
ranks = [
    int(ranking_df[ranking_df['Model'] == m]['Rank'].values[0])
    for m in MODELS
]

bars3 = ax3.bar(MODELS, scores, color=ATTACK_COLOR)

for bar, s, r in zip(bars3, scores, ranks):
    ax3.text(
        bar.get_x() + bar.get_width()/2,
        s + 0.01,
        f'#{r}\n{s:.3f}',
        ha='center',
        fontsize=9
    )

ax3.set_ylim(0, 0.65)
ax3.set_ylabel('Robustness Score', fontsize=10)
ax3.set_title('③ Robustness Score',
              fontsize=11, fontweight='bold', color=ACCENT_COLOR)
ax3.grid(True, axis='y', linestyle='--', alpha=0.3)

# ============================================================
# Panel 4: Per-Class ASR (PGD @ ε = 0.2)
# ============================================================
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor('white')

pgd_02_subset = per_class_df[
    (per_class_df['Attack'] == 'PGD') &
    (per_class_df['Epsilon'] == 0.2)
]

mini_matrix = np.array([
    [
        pgd_02_subset[pgd_02_subset['Category'] == cat][f'ASR_{m}'].values[0]
        for m in MODELS
    ]
    for cat in CATEGORIES
])

im4 = ax4.imshow(mini_matrix, cmap='Blues', vmin=0, vmax=100)

for i in range(len(CATEGORIES)):
    for j in range(len(MODELS)):
        v = mini_matrix[i, j]
        ax4.text(j, i, f'{v:.0f}%', ha='center', va='center',
                 fontsize=9, color='white' if v > 55 else 'black')

ax4.set_xticks(range(len(MODELS)))
ax4.set_yticks(range(len(CATEGORIES)))
ax4.set_xticklabels(MODELS, fontsize=10)
ax4.set_yticklabels(CATEGORIES, fontsize=10)
ax4.set_title('④ Per-Class ASR (PGD @ ε = 0.2)',
              fontsize=11, fontweight='bold', color=ACCENT_COLOR)

# ============================================================
# Panel 5: FPR Comparison (Baseline vs PGD)
# ============================================================
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor('white')

x5 = np.arange(len(models_ordered))
w5 = 0.35

ax5.bar(x5 - w5/2, fpr_base, w5,
        color=BASE_COLOR, label='Baseline')

ax5.bar(x5 + w5/2, fpr_pgd, w5,
        color=ATTACK_COLOR, label='After PGD')

ax5.set_xticks(x5)
ax5.set_xticklabels(models_ordered, fontsize=10)
ax5.set_ylim(0, 110)
ax5.set_ylabel('FPR (%)', fontsize=10)
ax5.set_title('⑤ False Positive Rate',
              fontsize=11, fontweight='bold', color=ACCENT_COLOR)
ax5.legend(frameon=False, fontsize=9)
ax5.grid(True, axis='y', linestyle='--', alpha=0.3)

# ============================================================
# Panel 6: Key Findings
# ============================================================
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('white')
ax6.axis('off')

findings = [
    ("Robustness Ranking:",
     "① XGB > RF >> MLP"),
    ("Worst-Case Impact:",
     "PGD attack dominates degradation"),
    ("Critical Classes:",
     "R2L & U2R collapse early"),
    ("Key Insight:",
     "Accuracy ≠ Robustness"),
]

y_pos = 0.95
for title, text in findings:
    ax6.text(0.05, y_pos, title,
             fontsize=10, fontweight='bold', color=ACCENT_COLOR)
    y_pos -= 0.06
    ax6.text(0.05, y_pos, text,
             fontsize=9, color='black')
    y_pos -= 0.1

ax6.set_title('⑥ Key Findings',
              fontsize=11, fontweight='bold', color=ACCENT_COLOR)

# ============================================================
# MAIN TITLE & SAVE
# ============================================================
fig.suptitle(
    'CISC 819 — Adversarial Robustness of ML-based NIDS\n'
    'Final Results Summary (NSL-KDD)',
    fontsize=15, fontweight='bold',
    y=1.01
)

fname_dashboard = f'{OUTPUT_DIR}/P5_Part7_Dashboard.png'
plt.savefig(fname_dashboard, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"Saved → {fname_dashboard}")

# ============================================================
# STEP 7.2: COMPLETE FIGURES INDEX
# Full list of all figures with captions — ready for report
# ============================================================

import os

print("\n" + "=" * 65)
print(" PHASE 5 — COMPLETE FIGURES INDEX")
print("=" * 65)

all_figures = [
    ("Part 2", "P5_Part2_ASR_Curve_RF.png",
     "Fig. 1 — RF Robustness Curve"),

    ("Part 2", "P5_Part2_ASR_Curve_XGB.png",
     "Fig. 2 — XGB Robustness Curve"),

    ("Part 2", "P5_Part2_ASR_Curve_MLP.png",
     "Fig. 3 — MLP Robustness Curve"),

    ("Part 2", "P5_Part2_ASR_Curves_Combined.png",
     "Fig. 1–3 (Combined) — All Models"),

    ("Part 3", "P5_Part3_Heatmap_ASR.png",
     "Fig. 4 — Per-Class Vulnerability Heatmap"),

    ("Part 3", "P5_Part3_Heatmap_Threshold.png",
     "Fig. 5 — Critical Threshold Heatmap"),

    ("Part 4", "P5_Part4_Feature_Normalized.png",
     "Fig. 6 — Feature Perturbation (Normalized)"),

    ("Part 4", "P5_Part4_Feature_Physical.png",
     "Fig. 7 — Feature Perturbation (Physical Units)"),

    ("Part 5", "P5_Part5_Radar_Combined.png",
     "Fig. 8 — Robustness Radar (Combined)"),
      ("Part 5", "P5_Part5_Radar_Individual.png",
     "Fig. 8b — Robustness Radar (Individual)"),

    ("Part 6", "P5_Part6_FPR_Bar.png",
     "Fig. 9 — FPR Before & After Attack"),

    ("Part 6", "P5_Part6_Scatter_AccVsASR.png",
     "Fig. 10 — Accuracy vs ASR Scatter"),

    ("Part 7", "P5_Part7_Dashboard.png",
     "Fig. 11 — Complete Summary Dashboard"),
]

for part, fname, desc in all_figures:
    full_path = f'{OUTPUT_DIR}/{fname}'
    exists = "✓" if os.path.exists(full_path) else "✗"

    print(f" {exists} [{part}] {desc}")
    print(f"     → {fname}")

print("\n" + "=" * 65)
print("  PHASE 5 — ALL PARTS COMPLETE!")
print("=" * 65)

print(f"\n  All figures saved in: {OUTPUT_DIR}/")
print(f"  Total figures: {len(all_figures)}")

print("\n  Ready for:")
print("   • Project Slides (April 13)")
print("   • Final Report (April 24)")

print("=" * 65)

