from pathlib import Path

fig_dir = Path("reports/figures")

files_to_remove = [
    "chapter7_cross_module_calibration.png",
    "chapter7_glm_coefficients_waterfall.png",
    "chapter7_shap_waterfall_demo.png",
    "chapter7_tree_importance_vs_eda.png"
]

for fname in files_to_remove:
    fpath = fig_dir / fname
    if fpath.exists():
        fpath.unlink()
        print(f"Removed: {fname}")
    else:
        print(f"Not found: {fname}")

# List remaining chapter 7 files
c7_files = sorted(list(fig_dir.glob("chapter7_*.png")))
print("\nRemaining Chapter 7 figures:")
for f in c7_files:
    print(f"  - {f.name}")
