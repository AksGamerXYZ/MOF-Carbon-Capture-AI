import json
import pandas as pd
import glob

files = glob.glob("/Users/atishmrk/OneDrive - Fort Bend Independent School District/bulk-dl-mofdb-version-dc8a0295db/*.json")

rows = []


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
                    "filename": file,
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

df.to_csv("hmof_co2_dataset.csv", index = False)