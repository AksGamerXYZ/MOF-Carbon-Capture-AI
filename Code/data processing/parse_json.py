import json
import pandas as pd
from pathlib import Path

# Code/data processing/parse_json.py -> parents[2] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = REPO_ROOT / "Data" / "bulk-dl-mofdb-version-dc8a0295db"
OUT_FILE = REPO_ROOT / "Data" / "hmof_co2_dataset.csv"

files = sorted(JSON_DIR.glob("*.json"))
if not files:
    raise FileNotFoundError(
        f"No .json files found in {JSON_DIR}. Download the MOFDB data into Data/ first."
    )

rows = []

for file in files:
    with open(file, "r") as f:
        data = json.load(f)

    for isotherm in data.get("isotherms", []):
        adsorbates = isotherm.get("adsorbates", [])

        if len(adsorbates) == 1 and adsorbates[0].get("formula") == "CO2":

            for point in isotherm.get("isotherm_data", []):
                row = {
                    "MOF": data.get("mofkey"),
                    "name": data.get("name"),
                    "id": data.get("id"),
                    "filename": file.name,
                    "surface_area": data.get("surface_area_m2g"),
                    "void_fraction": data.get("void_fraction"),
                    "LCD": data.get("lcd"),
                    "PLD": data.get("pld"),
                    "temperature": isotherm.get("temperature"),
                    "pressure": point.get("pressure"),
                    "CO2_adsorption": point.get("total_adsorption")
                }

                rows.append(row)

df = pd.DataFrame(rows)

print(df.shape)

df.to_csv(OUT_FILE, index=False)
