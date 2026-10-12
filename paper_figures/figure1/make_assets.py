"""Figure 1 assets (NMI resubmission, October 2026).

Produces the raster pieces that the pptxgenjs script places on the slide,
plus a JSON file with every number the slide needs. Nothing under the
original project is modified; the three PyVista assembly renders are copied
from claude_figure/revised_nature_3.

Outputs (assets/):
  lattice_input.png     17 x 17 cross-section of the corpus example lattice
  mol_toluidine.png     p-toluidine, skeletal, RDKit (Arial)
  mol_product.png       Ar-NH-p-tolyl product, skeletal, RDKit (Arial)
  yield_heatmap.png     measured yields, 4 ligands x 22 additives, one aryl
                        halide and one base (colour = physical quantity)
  assembly_*.png        copied 3D renders
  data/fig1_assets.json numbers used by make_fig1.js
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # .../reactor_genome
ASSETS = HERE / "assets"
DATA = HERE / "data"
ASSETS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

DPI = 600
GD = "#0072B2"      # material colour used by the 3D renders
GD_E = "#00527F"
FUEL = "#DFE3E7"
FUEL_E = "#B4BBC2"
GT_E = "#6B7480"
ARIAL = next((p for p in ["/Library/Fonts/Arial.ttf",
                          "/System/Library/Fonts/Supplemental/Arial.ttf"] if Path(p).exists()), None)
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42})

out = {}

# ------------------------------------------------------------ 3D renders (copied)
src = ROOT / "claude_figure/revised_nature_3"
for a, b in [("assembly2_input_trim.png", "assembly_input.png"),
             ("assembly2_output_trim.png", "assembly_output.png"),
             ("assembly2_fields_trim.png", "assembly_fields.png")]:
    shutil.copyfile(src / a, ASSETS / b)
out["fields"] = json.load(open(src / "fields_stats.json"))

# ------------------------------------------------------------ lattice cross-section
D = json.load(open(ROOT / "claude_figure/revised_fable1/fig1_data.json"))
grid = D["example"]["grid"]
assert len(grid) == 289
out["example"] = {"gd": grid.count("g"), "k": D["example"]["k"]}
out["dpo_best"] = {"gd": D["dpo_best"]["gd"]}

fig = plt.figure(figsize=(1.0, 1.0), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.5, 16.5)
ax.set_ylim(16.5, -0.5)
ax.set_aspect("equal")
ax.axis("off")
for i in tqdm(range(289), desc="lattice", leave=False):
    r, c = divmod(i, 17)
    s = grid[i]
    if s == "f":
        ax.add_patch(Circle((c, r), 0.40, fc=FUEL, ec=FUEL_E, lw=0.25))
    elif s == "g":
        ax.add_patch(Circle((c, r), 0.40, fc=GD, ec=GD_E, lw=0.25))
    else:
        ax.add_patch(Circle((c, r), 0.44, fc="white", ec=GT_E, lw=0.6))
fig.savefig(ASSETS / "lattice_input.png", dpi=DPI, transparent=True)
plt.close(fig)

# ------------------------------------------------------------ molecules (RDKit)
def draw_mol(smiles, name, w_px, h_px, labels=None):
    """Canvas sized 1:1 for a 600 dpi render of the 2x slide, so atom labels are
    117 px = 14 pt on the slide = 7 pt in print, and bonds are 0.6 pt in print."""
    mol = Chem.MolFromSmiles(smiles)
    for idx, lab in (labels or {}).items():
        mol.GetAtomWithIdx(idx).SetProp("atomLabel", lab)
    d = rdMolDraw2D.MolDraw2DCairo(w_px, h_px)
    o = d.drawOptions()
    o.setBackgroundColour((1, 1, 1, 0))     # transparent
    o.useBWAtomPalette()
    if ARIAL:
        o.fontFile = ARIAL
    o.padding = 0.06
    o.bondLineWidth = 10
    o.scaleBondWidth = False
    o.minFontSize = 117
    o.maxFontSize = 117
    o.additionalAtomLabelPadding = 0.10
    d.DrawMolecule(mol)
    d.FinishDrawing()
    (ASSETS / name).write_bytes(d.GetDrawingText())

draw_mol("Nc1ccc(C)cc1", "mol_toluidine.png", 768, 444)                       # 0.64 in print
draw_mol("[*]Nc1ccc(C)cc1", "mol_product.png", 1008, 456, labels={0: "Ar"})   # 0.84 in print

# ------------------------------------------------------------ measured-yield heatmap
bh = pd.read_csv(ROOT / "empirical/data/buchwald_hartwig.csv")
assert len(bh) == 3955
lig = list(dict.fromkeys(bh["Ligand"]))          # dataset order: XPhos, t-BuXPhos, t-BuBrettPhos, AdBrettPhos
add = list(dict.fromkeys(bh["Additive"]))
base = list(dict.fromkeys(bh["Base"]))
aryl = list(dict.fromkeys(bh["Aryl halide"]))
LIG_NAMES = ["XPhos", "t-BuXPhos", "t-BuBrettPhos", "AdBrettPhos"]
BASE_NAMES = ["P2Et", "BTMG", "MTBD"]
# one aryl halide and one base: the complete 4 x 22 slice with the widest yield range
best = None
for a in aryl:
    for b in base:
        s = bh[(bh["Aryl halide"] == a) & (bh["Base"] == b)]
        if len(s) < 88:
            continue
        rng = s["Output"].max() - s["Output"].min()
        if best is None or rng > best[0]:
            best = (rng, a, b)
_, A_SEL, B_SEL = best
s = bh[(bh["Aryl halide"] == A_SEL) & (bh["Base"] == B_SEL)]
M = np.full((4, 22), np.nan)
for _, r in s.iterrows():
    M[lig.index(r["Ligand"]), add.index(r["Additive"])] = r["Output"]
assert not np.isnan(M).any()
out["yield"] = {"aryl_halide_smiles": A_SEL, "base": BASE_NAMES[base.index(B_SEL)],
                "ligands": LIG_NAMES, "n_additives": 22, "vmin": 0.0, "vmax": 100.0,
                "max_observed": float(np.nanmax(M)), "min_observed": float(np.nanmin(M))}

cw = 0.062                                   # inch per cell at print size
fig = plt.figure(figsize=(22 * cw, 4 * cw), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(M, cmap="cividis", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
ax.set_xticks(np.arange(-0.5, 22, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
ax.grid(which="minor", color="white", lw=0.35)
ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.savefig(ASSETS / "yield_heatmap.png", dpi=DPI, transparent=True)
plt.close(fig)

# colour strip for the yield bar (cividis, 0 to 100)
fig = plt.figure(figsize=(1.0, 0.08), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(np.linspace(0, 1, 256)[None, :], cmap="cividis", aspect="auto")
ax.axis("off")
fig.savefig(ASSETS / "cb_cividis.png", dpi=DPI, transparent=True)
plt.close(fig)
for name in ["viridis", "magma"]:
    fig = plt.figure(figsize=(1.0, 0.08), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=name, aspect="auto")
    ax.axis("off")
    fig.savefig(ASSETS / f"cb_{name}.png", dpi=DPI, transparent=True)
    plt.close(fig)

# ------------------------------------------------------------ aligned-policy inventory histogram
t6 = pd.read_csv(ROOT / "professor/Test/t6_generation_yield/t6_samples_v2.csv")
assert len(t6) == 10000 and t6.grid.str.count("g").eq(t6.gd_count).all()
cnt = t6.gd_count.value_counts().sort_index()
hist = {int(k): int(v) for k, v in cnt.items()}
out["policy_inventory"] = {"n": 10000, "mean": float(t6.gd_count.mean()), "sd": float(t6.gd_count.std(ddof=1)),
                           "max": int(t6.gd_count.max()), "hist": hist}

# target-window hits of the two random procedures (all inside 28-35) and the aligned subsample
ev = pd.read_csv(ROOT / "professor/Test/t6_generation_yield/t6_openmc_subsample.csv")
hit = (ev.k_eff - 1.05).abs() <= 0.005
out["aligned_hits"] = {"n": int(len(ev)), "hits": int(hit.sum()),
                       "hit_inventory_min": int(ev.loc[hit, "gd"].min()), "hit_inventory_max": int(ev.loc[hit, "gd"].max())}
uni = pd.concat([pd.read_csv(ROOT / f"alam_additional_expriment/results/uninformed_random_seed{s}_results.csv") for s in range(5)])
gcol = "g_count" if "g_count" in uni else [c for c in uni.columns if "count" in c][0]
uh = (uni.k_eff - 1.05).abs() <= 0.005
out["uniform_hits"] = {"n": int(len(uni)), "hits": int(uh.sum()), "min": int(uni.loc[uh, gcol].min()), "max": int(uni.loc[uh, gcol].max())}
inf = pd.concat([pd.read_csv(ROOT / "professor/Test/t3_random_search/random_search_results.csv")] +
                [pd.read_csv(ROOT / f"professor/Test/t3_random_search/oracle_seed{s}_results.csv") for s in range(1, 5)])
ih = (inf.k_eff - 1.05).abs() <= 0.005
out["informed_hits"] = {"n": int(len(inf)), "hits": int(ih.sum()), "min": int(inf.loc[ih, "g_count"].min()), "max": int(inf.loc[ih, "g_count"].max())}
nop = [pd.read_csv(ROOT / f"alam_additional_expriment/results/no_penalty_dpo_seed{s}_results.csv").chosen_g_count.tail(100).mean() for s in range(5)]
out["no_penalty_final100"] = [float(round(v, 2)) for v in nop]

# ------------------------------------------------------------ chemistry exact probabilities
f4 = pd.read_csv(ROOT / "NMI_submit/1st appeal/figure/fig4_chemistry_plotted_values.csv")
q = f4[(f4.regime == "single") & (f4.metric == "expansion_mass")].set_index("procedure")["mean"]
e = f4[(f4.regime == "single") & (f4.metric == "expected_abs_error")].set_index("procedure")["mean"]
out["chem"] = {"off_corpus": {k: float(q[k]) for k in ["sft_base", "dpo", "grpo", "random_full"]},
               "error": {k: float(e[k]) for k in ["sft_base", "dpo", "grpo", "random_full"]}}

json.dump(out, open(DATA / "fig1_assets.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "policy_inventory"}, indent=1)[:1500])
print("policy inventory mean/sd/max:", out["policy_inventory"]["mean"], out["policy_inventory"]["sd"], out["policy_inventory"]["max"])
