"""Generate figures for Chapter 9 — Synthese des resultats."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'DejaVu Sans',
})

# ============================================================
# FIGURE 9.1 — Gains relatifs par module
# ============================================================
# Chaque module est represente par son gain relatif (%) :
#   Tarification    : Gini    0.278 (GLM)       -> 0.290 (CANN)       = +4.2%
#   Provisionnement : couv.   74.4%  (Mack)     -> 91.9%  (Conformal) = +23.5%
#   Fraude          : AUC-ROC 0.527  (IsolF.)   -> 0.815  (RandomFor) = +54.7%
fig1, ax1 = plt.subplots(figsize=(10, 5))

modules = ['Tarification\n(Gini)', 'Provisionnement\n(Couverture 90%)', 'Fraude\n(AUC-ROC)']
gains = [4.2, 23.5, 54.7]                 # gains relatifs (%)

y = np.arange(len(modules))

bars3 = ax1.barh(y, gains, height=0.55, color='#1565C0', edgecolor='#0D47A1',
                 linewidth=0.8, label='Gain relatif du modele final vs baseline')

# Gain labels
gain_labels = ['GLM 0.278\n-> CANN 0.290', 'Mack 74.4%\n-> Conformal 91.9%',
               'Isol. Forest 0.527\n-> Random Forest 0.815']
for i, (g, gl) in enumerate(zip(gains, gain_labels)):
    ax1.text(g + 1, i, gl, va='center', fontsize=9, color='#0D47A1',
             fontweight='bold')

for i, g in enumerate(gains):
    ax1.text(g - 1, i, f'+{g:.1f}%', va='center', ha='right',
             fontsize=11, fontweight='bold', color='white')

ax1.set_yticks(y)
ax1.set_yticklabels(modules, fontsize=11)
ax1.set_xlabel('Gain relatif du modele final par rapport a la baseline (%)')
ax1.set_title('Figure 9.1  --  Gains relatifs par module : baseline vs modele final',
              fontsize=13, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=10)
ax1.set_xlim(0, 62)

plt.tight_layout()
fig1.savefig('reports/figures/chapter9_01_gains_comparatifs.png', dpi=150, bbox_inches='tight')
print('Figure 9.1 saved')
plt.close()

# ============================================================
# FIGURE 9.2 — Couverture des intervalles
# ============================================================
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={'width_ratios': [1.2, 1]})

# Left: Bar chart of coverage
methods = ['Mack\nChain-Ladder', 'Mack +\nConformal', 'NGBoost\nSeverity']
coverages = [74.4, 91.9, 90.58]
colors = ['#E53935', '#1565C0', '#2E7D32']
target = 90.0

bars = ax2a.bar(methods, coverages, color=colors, edgecolor='white',
                width=0.6, linewidth=1.5)
ax2a.axhline(y=target, color='#FF6F00', linestyle='--', linewidth=2,
             label='Cible 90%')
ax2a.set_ylabel('Couverture empirique (%)')
ax2a.set_title('Couverture des intervalles de confiance',
               fontsize=12, fontweight='bold')
ax2a.set_ylim(60, 100)
ax2a.legend(fontsize=10)

for bar, cov in zip(bars, coverages):
    ax2a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
              f'{cov}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Right: Interval width comparison
width_labels = ['Mack seul', 'Mack + Conformal']
width_vals = [4503, 10947]
width_colors = ['#E53935', '#1565C0']

bars_w = ax2b.barh(width_labels, width_vals, color=width_colors,
                    height=0.5, edgecolor='white')
ax2b.set_xlabel('Largeur moyenne des intervalles ($)')
ax2b.set_title('Largeur des intervalles', fontsize=12, fontweight='bold')

for bar, val in zip(bars_w, width_vals):
    ax2b.text(val + 200, bar.get_y() + bar.get_height()/2,
              f'${val:,.0f}', va='center', fontweight='bold', fontsize=11)

ax2b.annotate('+145% (cout de la garantie)', xy=(10947, 1),
              xytext=(7000, 1.35), fontsize=9, ha='center', color='#D84315',
              arrowprops=dict(arrowstyle='->', color='#D84315', lw=1.5))

fig2.suptitle('Figure 9.2  --  Couverture et largeur des intervalles de confiance',
              fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig2.savefig('reports/figures/chapter9_02_couverture_intervalles.png',
             dpi=150, bbox_inches='tight')
print('Figure 9.2 saved')
plt.close()

# ============================================================
# FIGURE 9.3 — Guide de decision
# ============================================================
fig3, ax3 = plt.subplots(figsize=(11, 7))
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')

ax3.text(5, 9.5, 'Figure 9.3  --  Guide de decision : quand utiliser le ML ?',
         ha='center', va='center', fontsize=14, fontweight='bold')

# Axes
ax3.annotate('', xy=(9.5, 0.3), xytext=(0.5, 0.3),
             arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
ax3.annotate('', xy=(0.5, 9.2), xytext=(0.5, 0.3),
             arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

ax3.text(5, 0.05, 'Complexite du signal a capter',
         ha='center', fontsize=11, fontweight='bold', color='#333')
ax3.text(0.05, 5, 'Criticite de\nl\'application',
         ha='center', va='center', fontsize=11, fontweight='bold',
         color='#333', rotation=90)

# Quadrant labels
ax3.text(2.5, 7, 'METHODE\nCLASSIQUE\nSUFFISANTE',
         ha='center', va='center', fontsize=11, fontweight='bold', color='#2E7D32',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                   edgecolor='#2E7D32', alpha=0.8))
ax3.text(7.5, 7, 'ML AVEC\nGARANTIE\nD\'INTERPRETABILITE',
         ha='center', va='center', fontsize=11, fontweight='bold', color='#E65100',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0',
                   edgecolor='#E65100', alpha=0.8))
ax3.text(2.5, 2.5, 'SIMPLIFIER\nAVANT DE COMPLEXIFIER',
         ha='center', va='center', fontsize=10, fontweight='bold', color='#555',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                   edgecolor='#9E9E9E', alpha=0.8))
ax3.text(7.5, 2.5, 'ML JUSTIFIE\nSANS EXCUSE',
         ha='center', va='center', fontsize=11, fontweight='bold', color='#1565C0',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD',
                   edgecolor='#1565C0', alpha=0.8))

# Examples
examples = [
    (3.5, 8.2, 'GLM Tarification', '#2E7D32', 10),
    (6.5, 8.5, 'Mack + Conformal', '#E65100', 10),
    (8, 4.5, 'CANN (+7% Gini)', '#E65100', 9),
    (8, 2, 'Random Forest\nFraude', '#1565C0', 9),
    (2, 3.5, 'Isolation Forest\n(AUC 0.527)', '#9E9E9E', 8),
    (5, 1.5, 'Deep Triangle\n(ratio instable)', '#9E9E9E', 8),
]

for x, y, text, color, size in examples:
    ax3.plot(x, y, 'o', color=color, markersize=8, zorder=5)
    ax3.text(x + 0.15, y, text, fontsize=size, color=color, va='center',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor=color, alpha=0.9))

plt.tight_layout()
fig3.savefig('reports/figures/chapter9_03_guide_decision.png',
             dpi=150, bbox_inches='tight')
print('Figure 9.3 saved')
plt.close()

print('All 3 figures generated successfully.')
