import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from pymatgen.core import Element, Structure
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

# pymatgen warns on every P1 CIF; keep the output readable
warnings.filterwarnings("ignore")

# Code/data processing/parse_cif_pytorch.py -> parents[2] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CIF_DIR = REPO_ROOT / "Data" / "bulk-dl-mofdb-version-dc8a0295db"
CO2_FILE = REPO_ROOT / "Data" / "hmof_co2_dataset.csv"  # made by parse_json.py
GRAPH_DIR = REPO_ROOT / "Data" / "hmof_graphs"

# CGCNN-style graph: each atom is linked to its nearest neighbors (periodic
# images included) inside a cutoff sphere
RADIUS = 8.0  # Angstrom
MAX_NBR = 12

# hMOF-0-(id_15339).cif -> name "hMOF-0", id 15339
NAME_RE = re.compile(r"^(?P<name>.+)-\(id_(?P<id>\d+)\)\.cif$")

# node feature vector, scaled to roughly 0-1:
# electronegativity, atomic radius, atomic mass, group, period, is_metal
ELEMENT_SCALES = np.array([4.0, 3.0, 100.0, 18.0, 7.0, 1.0], dtype=np.float32)


@lru_cache(maxsize=None)
def element_features(symbol):
    el = Element(symbol)
    raw = [el.X, el.atomic_radius, el.atomic_mass, el.group, el.row, el.is_metal]
    # pymatgen gives None / nan for some properties (e.g. noble gas electronegativity)
    vec = np.array([0.0 if v is None else float(v) for v in raw], dtype=np.float32)
    return np.nan_to_num(vec) / ELEMENT_SCALES


def featurize(path):
    """Return (id, arrays, error) for one CIF; arrays/error is None on the other's success."""
    try:
        m = NAME_RE.match(path.name)
        s = Structure.from_file(path)
        n = len(s)

        # nearest MAX_NBR neighbors per atom, nearest first
        center, nbr, _, dist = s.get_neighbor_list(RADIUS)
        order = np.lexsort((dist, center))
        center, nbr, dist = center[order], nbr[order], dist[order]
        starts = np.r_[0, np.flatnonzero(np.diff(center)) + 1]
        counts = np.diff(np.r_[starts, len(center)])
        rank = np.arange(len(center)) - np.repeat(starts, counts)
        keep = rank < MAX_NBR

        arrays = {
            "z": np.array([site.specie.Z for site in s], dtype=np.int64),
            "x": np.stack([element_features(site.specie.symbol) for site in s]),
            "pos": s.lattice.get_cartesian_coords(s.frac_coords % 1.0).astype(np.float32),
            "cell": s.lattice.matrix.astype(np.float32),
            # messages flow neighbor -> center atom
            "edge_index": np.stack([nbr[keep], center[keep]]).astype(np.int64),
            "edge_attr": dist[keep].astype(np.float32)[:, None],
            "density": float(s.density),  # g/cm^3
            "name": m["name"],
        }
        return int(m["id"]), arrays, None
    except Exception as e:
        return None, None, f"{path.name}: {type(e).__name__}: {e}"


class HMOFGraphDataset(InMemoryDataset):
    """One PyG graph per hMOF, with its CO2 uptake at the 5 simulated pressures.

    Per graph:
      x           [N, 6]   scaled element properties (see ELEMENT_SCALES)
      z           [N]      atomic number, for an nn.Embedding
      pos         [N, 3]   Cartesian coordinates, wrapped into the cell
      edge_index  [2, E]   neighbor -> center atom
      edge_attr   [E, 1]   interatomic distance in Angstrom (expand it in the model)
      cell        [1, 3, 3] lattice vectors
      u           [1, 5]   surface_area (m2/g), void_fraction, LCD, PLD, density (g/cm3)
      pressure    [1, 5]   bar, ascending
      y           [1, 5]   CO2 uptake in mol/kg at `pressure`, 298 K
      mof_id      [1]      MOFDB id, use it to split by MOF
      name        str      e.g. "hMOF-0"
    """

    def __init__(self, root=GRAPH_DIR, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # raw data is the MOFDB download in Data/, nothing to fetch

    @property
    def processed_file_names(self):
        return ["hmof_co2_graphs.pt"]

    def process(self):
        if not CO2_FILE.exists():
            raise FileNotFoundError(f"{CO2_FILE} not found. Run parse_json.py first.")

        co2 = pd.read_csv(
            CO2_FILE,
            usecols=["id", "surface_area", "void_fraction", "LCD", "PLD", "pressure", "CO2_adsorption"],
        )
        uptake = co2.pivot(index="id", columns="pressure", values="CO2_adsorption").dropna()
        pressure = uptake.columns.to_numpy(dtype=np.float32)
        descriptors = co2.groupby("id")[["surface_area", "void_fraction", "LCD", "PLD"]].first()
        descriptors = descriptors.reindex(uptake.index).to_numpy(dtype=np.float32)
        row_of = {mof_id: i for i, mof_id in enumerate(uptake.index)}
        y_all = uptake.to_numpy(dtype=np.float32)

        files = sorted(
            p for p in CIF_DIR.glob("*.cif")
            if (m := NAME_RE.match(p.name)) and int(m["id"]) in row_of
        )
        if not files:
            raise FileNotFoundError(
                f"No matching .cif files found in {CIF_DIR}. Download the MOFDB data into Data/ first."
            )

        data_list, errors = [], []
        # each worker imports pymatgen and torch, so don't spawn one per core
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            for mof_id, a, err in tqdm(pool.map(featurize, files, chunksize=32), total=len(files)):
                if err:
                    errors.append(err)
                    continue
                i = row_of[mof_id]
                u = np.append(descriptors[i], a["density"]).astype(np.float32)
                data_list.append(Data(
                    x=torch.from_numpy(a["x"]),
                    z=torch.from_numpy(a["z"]),
                    pos=torch.from_numpy(a["pos"]),
                    edge_index=torch.from_numpy(a["edge_index"]),
                    edge_attr=torch.from_numpy(a["edge_attr"]),
                    cell=torch.from_numpy(a["cell"]).unsqueeze(0),
                    u=torch.from_numpy(u).unsqueeze(0),
                    pressure=torch.from_numpy(pressure).unsqueeze(0),
                    y=torch.from_numpy(y_all[i]).unsqueeze(0),
                    mof_id=torch.tensor([mof_id]),
                    name=a["name"],
                ))

        if errors:
            print(f"{len(errors)} CIF files failed, first few:")
            for e in errors[:5]:
                print("  ", e)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])


def main():
    # built once and cached in GRAPH_DIR/processed; delete that folder to rebuild
    dataset = HMOFGraphDataset(GRAPH_DIR)
    print(dataset)
    print(dataset[0])
    print(f"{dataset._data.num_nodes} atoms and {dataset._data.num_edges} edges in total")


if __name__ == "__main__":
    main()
