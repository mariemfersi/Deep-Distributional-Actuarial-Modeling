import json
from pathlib import Path

nb_path = Path("notebooks/05_fraud.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scratch/nb_fraud_output.txt", "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb["cells"]):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        out.write(f"--- CELL {i} ({cell_type}) ---\n")
        out.write(source + "\n\n")

print("Wrote notebook to scratch/nb_fraud_output.txt")
