"""Generate figures for Chapter 10 — Conclusion generale."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': False,
    'font.family': 'DejaVu Sans',
})

# Palette
C_BASE  = '#B0BEC5'
C_BASE2 = '#78909C'
C_FINAL = '#1565C0'
C_GREEN = '#2E7D32'
C_ORANGE = '#E65100'
C_RED   = '#C62828'
C_BG    = '#F5F5F5'

# ============================================================
# FIGURE 10.1 — Synthese des resultats finaux
# ============================================================
# Trois panneaux compacts : un par module, montrant baseline vs final
fig1, axes = plt.subplots(1, 3, figsize=(15, 5),
                           gridspec_kw={'wspace': 0.35})

# --- Panel A : Tarification ---
ax_a = axes[0]
ax_a.set_xlim(0, 1)
ax_a.set_ylim(0, 1)
ax_a.axis('off')
ax_a.set_facecolor('#E3F2FD')
for spine in ax_a.spines.values():
    spine.set_edgecolor('#1565C0')
    spine.set_linewidth(1.5)

ax_a.text(0.5, 0.95, 'TARIFICATION', ha='center', va='top',
          fontsize=12, fontweight='bold', color=C_FINAL)
ax_a.text(0.5, 0.88, 'Gini index (test set)', ha='center', va='top',
          fontsize=9, color='#555')

# Bars
bar_data = [
    ('GLM Poisson', 0.278, C_BASE),
    ('CANN cible', 0.290, C_FINAL),
]
for i, (label, val, color) in enumerate(bar_data):
    y = 0.72 - i * 0.22
    ax_a.barh(y, val / 0.35, height=0.14, left=0.02, color=color,
              edgecolor='white', linewidth=1)
    ax_a.text(0.02 + val / 0.35 + 0.02, y, f'{val:.3f}',
              va='center', fontsize=10, fontweight='bold', color=color)
    ax_a.text(0.01, y, label, va='center', ha='right', fontsize=9,
              color='#333')

ax_a.annotate('', xy=(0.02 + 0.290/0.35, 0.72 - 0.22),
              xytext=(0.02 + 0.278/0.35, 0.72),
              arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.8))
ax_a.text(0.02 + 0.284/0.35, 0.72 - 0.11, '+4.2%\nrelatif',
          ha='center', va='center', fontsize=9, fontweight='bold',
          color=C_GREEN)

ax_a.text(0.5, 0.22, 'Deviance/obs : 0.3179 → 0.3130 (−1.56%)',
          ha='center', va='center', fontsize=9, color='#555',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#90CAF9', alpha=0.9))
ax_a.text(0.5, 0.08, 'NGBoost sev. : LL −2.34, CRPS 0.45\n'
          'Couv. 90% = 90.58%',
          ha='center', va='center', fontsize=8, color='#666')

# --- Panel B : Provisionnement ---
ax_b = axes[1]
ax_b.set_xlim(0, 1)
ax_b.set_ylim(0, 1)
ax_b.axis('off')
ax_b.set_facecolor('#E8F5E9')
for spine in ax_b.spines.values():
    spine.set_edgecolor(C_GREEN)
    spine.set_linewidth(1.5)

ax_b.text(0.5, 0.95, 'PROVISIONNEMENT', ha='center', va='top',
          fontsize=12, fontweight='bold', color=C_GREEN)
ax_b.text(0.5, 0.88, 'Couverture empirique (cible 90%)', ha='center',
          va='top', fontsize=9, color='#555')

# Target line at 90%
target_x = 0.02 + 0.90 * 0.9  # scaled to panel
bar_data2 = [
    ('Mack seul', 0.744, C_RED),
    ('+ Conforme', 0.919, C_GREEN),
]
for i, (label, val, color) in enumerate(bar_data2):
    y = 0.72 - i * 0.22
    ax_b.barh(y, val * 0.9, height=0.14, left=0.02, color=color,
              edgecolor='white', linewidth=1)
    ax_b.text(0.02 + val * 0.9 + 0.02, y, f'{val:.1%}',
              va='center', fontsize=10, fontweight='bold', color=color)
    ax_b.text(0.01, y, label, va='center', ha='right', fontsize=9,
              color='#333')

# Target line
ax_b.axvline(x=0.02 + 0.90 * 0.9, color='#FF6F00', linestyle='--',
             linewidth=1.5, zorder=5)
ax_b.text(0.02 + 0.90 * 0.9, 0.38, '90%', ha='center', fontsize=8,
          color='#FF6F00', fontweight='bold')

ax_b.annotate('', xy=(0.02 + 0.919 * 0.9, 0.72 - 0.22),
              xytext=(0.02 + 0.744 * 0.9, 0.72),
              arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.8))
ax_b.text(0.02 + 0.832 * 0.9, 0.72 - 0.11, '+23.5%\nrelatif',
          ha='center', va='center', fontsize=9, fontweight='bold',
          color=C_GREEN)

ax_b.text(0.5, 0.22, 'Mack seul : 74.4% (sous-couverture)\n'
          'Conforme : q̂ = 4.00 (vs z = 1.645)',
          ha='center', va='center', fontsize=9, color='#555',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#A5D6A7', alpha=0.9))
ax_b.text(0.5, 0.08, 'Deep Triangle GRU : démo technique\n'
          '(instable, ratio 0.57–2.65)',
          ha='center', va='center', fontsize=8, color='#666')

# --- Panel C : Fraude ---
ax_c = axes[2]
ax_c.set_xlim(0, 1)
ax_c.set_ylim(0, 1)
ax_c.axis('off')
ax_c.set_facecolor('#FFF3E0')
for spine in ax_c.spines.values():
    spine.set_edgecolor(C_ORANGE)
    spine.set_linewidth(1.5)

ax_c.text(0.5, 0.95, 'DÉTECTION DE FRAUDE', ha='center', va='top',
          fontsize=12, fontweight='bold', color=C_ORANGE)
ax_c.text(0.5, 0.88, 'AUC-ROC (test set)', ha='center', va='top',
          fontsize=9, color='#555')

bar_data3 = [
    ('Isol. Forest', 0.527, C_BASE),
    ('Random Forest', 0.815, C_ORANGE),
]
for i, (label, val, color) in enumerate(bar_data3):
    y = 0.72 - i * 0.22
    ax_c.barh(y, val, height=0.14, left=0.02, color=color,
              edgecolor='white', linewidth=1)
    ax_c.text(val + 0.02 + 0.02, y, f'{val:.3f}',
              va='center', fontsize=10, fontweight='bold', color=color)
    ax_c.text(0.01, y, label, va='center', ha='right', fontsize=9,
              color='#333')

# Random reference line
ax_c.axvline(x=0.02 + 0.50, color='#999', linestyle=':', linewidth=1.2)
ax_c.text(0.02 + 0.50, 0.38, 'hasard', ha='center', fontsize=8,
          color='#999')

ax_c.annotate('', xy=(0.02 + 0.815, 0.72 - 0.22),
              xytext=(0.02 + 0.527, 0.72),
              arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.8))
ax_c.text(0.02 + 0.671, 0.72 - 0.11, '+54.7%\nrelatif',
          ha='center', va='center', fontsize=9, fontweight='bold',
          color=C_GREEN)

ax_c.text(0.5, 0.22, 'Fault (0.241), BasePolicy (0.173)\n'
          'PolicyType (0.100) variables clés',
          ha='center', va='center', fontsize=9, color='#555',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#FFCC80', alpha=0.9))
ax_c.text(0.5, 0.08, '4 tentatives GNN avortées\n'
          '(homophilie ≤ référence aléatoire)',
          ha='center', va='center', fontsize=8, color='#666')

fig1.suptitle('Figure 10.1  —  Résultats finaux par module : baseline classique vs modèle retenu',
              fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig1.savefig('reports/figures/chapter10_01_resultats_finaux.png',
             dpi=150, bbox_inches='tight')
print('Figure 10.1 saved')
plt.close()

# ============================================================
# FIGURE 10.2 — Demarche methodologique en 5 etapes
# ============================================================
fig2, ax2 = plt.subplots(figsize=(14, 4.5))
ax2.set_xlim(-0.5, 5.5)
ax2.set_ylim(-1.5, 4.0)
ax2.axis('off')

ax2.text(2.5, 3.7, 'Figure 10.2  —  Démarche méthodologique du projet',
         ha='center', va='center', fontsize=13, fontweight='bold')

stages = [
    ('Baseline\nFirst',   'GLM Poisson\nMack Chain-Ladder\nIsolation Forest',
     '#B0BEC5', '#455A64'),
    ('ML\nEnhancement',   'CANN interaction\nConformal Prediction\nRandom Forest',
     '#BBDEFB', C_FINAL),
    ('Empirical\nValidation', 'Gini 0.278→0.290\nCouv. 74→92%\nAUC 0.53→0.82',
     '#C8E6C9', C_GREEN),
    ('Explainability',    'SHAP pricing\nSHAP fraude\nFeature importance',
     '#FFF9C4', '#F57F17'),
    ('Honest\nReporting', 'CANN générique ✗\nDeep Triangle ≈\nGNN ✗✗✗✗',
     '#FFCCBC', C_RED),
]

for i, (title, detail, bg, fg) in enumerate(stages):
    x = i
    # Box
    rect = mpatches.FancyBboxPatch((x - 0.42, -1.1), 0.84, 4.2,
                                    boxstyle="round,pad=0.1",
                                    facecolor=bg, edgecolor=fg,
                                    linewidth=1.5, zorder=1)
    ax2.add_patch(rect)

    # Stage number
    ax2.plot(x, 2.7, 'o', color=fg, markersize=22, zorder=3)
    ax2.text(x, 2.7, str(i + 1), ha='center', va='center',
             fontsize=13, fontweight='bold', color='white', zorder=4)

    # Title
    ax2.text(x, 2.0, title, ha='center', va='center',
             fontsize=10, fontweight='bold', color=fg, zorder=3)

    # Detail lines
    ax2.text(x, 0.7, detail, ha='center', va='center',
             fontsize=8, color='#333', linespacing=1.4, zorder=3)

    # Arrow between stages
    if i < 4:
        ax2.annotate('', xy=(x + 0.48, 1.0), xytext=(x + 0.42, 1.0),
                      arrowprops=dict(arrowstyle='->', color='#666',
                                      lw=2, mutation_scale=15),
                      zorder=2)

plt.tight_layout()
fig2.savefig('reports/figures/chapter10_02_demarche_methodologique.png',
             dpi=150, bbox_inches='tight')
print('Figure 10.2 saved')
plt.close()

print('All 2 figures generated successfully.')
