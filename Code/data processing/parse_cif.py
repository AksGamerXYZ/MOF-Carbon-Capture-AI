import os
import re
import warnings
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from pymatgen.core import Structure
from tqdm import tqdm

# pymatgen warns on every P1 CIF; keep the output readable
warnings.filterwarnings("ignore")

# Code/data processing/parse_cif.py -> parents[2] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CIF_DIR = REPO_ROOT / "Data" / "bulk-dl-mofdb-version-dc8a0295db"
OUT_FILE = REPO_ROOT / "Data" / "hmof_cif_features.csv"

# hMOF-0-(id_15339).cif -> name "hMOF-0", id 15339
NAME_RE = re.compile(r"^(?P<name>.+)-\(id_(?P<id>\d+)\)\.cif$")


def featurize(path):
    """Return (row, error) for one CIF; exactly one of them is None."""
    try:
        m = NAME_RE.match(path.name)
        s = Structure.from_file(path)
        comp = s.composition
        n_atoms = len(s)
        n_metal = int(round(sum(amt for el, amt in comp.items() if el.is_metal)))

        a, b, c = s.lattice.abc
        alpha, beta, gamma = s.lattice.angles

        row = {
            "id": int(m["id"]) if m else None,
            "name": m["name"] if m else None,
            "cif_filename": path.name,
            "n_atoms": n_atoms,
            "n_elements": len(comp.elements),
            "cell_a": a,
            "cell_b": b,
            "cell_c": c,
            "cell_alpha": alpha,
            "cell_beta": beta,
            "cell_gamma": gamma,
            "cell_volume": s.volume,
            "volume_per_atom": s.volume / n_atoms,
            "density": float(s.density),  # g/cm^3
            "cell_mass": float(comp.weight),  # amu
            "avg_electronegativity": float(comp.average_electroneg),
            "n_metal_atoms": n_metal,
            "metal_fraction": n_metal / n_atoms,
        }
        for el, amt in comp.items():
            row[f"frac_{el.symbol}"] = amt / n_atoms
        return row, None
    except Exception as e:
        return None, f"{path.name}: {type(e).__name__}: {e}"


def main():
    files = sorted(CIF_DIR.glob("*.cif"))
    if not files:
        raise FileNotFoundError(
            f"No .cif files found in {CIF_DIR}. Download the MOFDB data into Data/ first."
        )

    rows, errors = [], []
    # each worker imports pymatgen (~300 MB), so don't spawn one per core
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        for row, err in tqdm(pool.map(featurize, files, chunksize=64), total=len(files)):
            if err:
                errors.append(err)
            else:
                rows.append(row)

    df = pd.DataFrame(rows)

    # an element absent from a MOF is a 0 fraction, not missing
    frac_cols = sorted(c for c in df.columns if c.startswith("frac_"))
    df[frac_cols] = df[frac_cols].fillna(0.0)
    df = df[[c for c in df.columns if c not in frac_cols] + frac_cols]

    print(df.shape)
    if errors:
        print(f"{len(errors)} CIF files failed, first few:")
        for e in errors[:5]:
            print("  ", e)

    df.to_csv(OUT_FILE, index=False)


if __name__ == "__main__":
    main()
