import json
from pathlib import Path

nb_path = Path("notebooks/04_reserving_mack.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    cell_type = cell.get("cell_type")
    source = "".join(cell.get("source", []))
    print(f"--- CELL {i} ({cell_type}) ---")
    print(source[:500])
    print("\n")
