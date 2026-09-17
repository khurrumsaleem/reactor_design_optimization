"""Depletion assessment of selected layouts (Methods, "Depletion assessment of
selected layouts"; Fig. 5 in the submitted manuscript, fig_depletion.pdf).

Depletes one fixed 17x17 layout in the same two-dimensional reflective
geometry and beginning-of-life material definitions used for transport
evaluation. Every fuel and Gd-bearing pin is an independent depletable
material (volume = pin cross-section x unit axial length). OpenMC transport is
coupled to the 228-nuclide simplified CASL PWR chain with the CE/CM
predictor-corrector integrator at a constant specific power of
35 W per g initial heavy metal, from 0 to 50 MWd/kgHM over 26 intervals
(27 burnup nodes). Each transport uses 20,000 particles per batch, 10 inactive
and 20 active batches.

After the trajectory, the endpoint composition is re-evaluated with every
remaining Gd isotope removed while all other nuclide densities are kept, to
give the residual Gd effect dk_Gd = k(endpoint, Gd removed) - k(endpoint).
The last downward crossing of k_inf = 1 is linearly interpolated between the
bracketing nodes. These are infinite-lattice metrics, not operating-reactor
cycle lengths; no thermal-hydraulic feedback; time-step convergence was not
evaluated. Depletion quantities never entered any alignment reward.

Layouts: DPO / GRPO / GA best-of-five-seeds and the symmetric 16-Gd and 24-Gd
references, read from selected_layouts.json (written by
analysis/first_passage.py, or taken from the archival data package).

Usage:
    python fetch_chain.py                                  # once
    python run_depletion.py --case DPO
    python run_depletion.py --case REF24 --layouts /path/selected_layouts.json
    python run_depletion.py --case DPO --smoke             # quick API check only
Outputs (in --output-root/<case>/): protocol.json, depletion_results.h5,
trajectory.csv, endpoint_without_residual_gd/, complete.json. Resumable at the
last completed burnup node.
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENMC_CROSS_SECTIONS',
                      os.path.expanduser('~/openmc_data/endfb-vii.1-hdf5/cross_sections.xml'))
import numpy as np  # noqa: E402
import openmc  # noqa: E402
import openmc.deplete  # noqa: E402
import openmc.deplete.pool  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'oracle'))
import eval_lib as E  # noqa: E402

openmc.deplete.pool.NUM_PROCESSES = int(os.environ.get('DEPLETION_PROCESSES', '2'))
CHAIN = HERE / 'chain_casl_pwr.xml'
DATA_ROOT = Path(os.environ.get('DATA_ROOT', HERE.parents[1] / 'data'))
BURNUPS = [0, .1, .25, .5, 1, 2, 3, 4, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25,
           27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50]
POWER_DENSITY = 35.0          # W per g initial heavy metal
PARTICLES, BATCHES, INACTIVE = 20000, 30, 10
SEED_TRAJECTORY, SEED_ENDPOINT = 420001, 420002


def make_model(gridstr):
    openmc.reset_auto_ids()
    fuel, gd, zirc, water = E.create_materials()
    grid = np.array([{'f': 0, 'g': 1, 'c': 2}[s] for s in gridstr]).reshape(17, 17)
    geometry = E.create_assembly(grid, fuel, gd, zirc, water)
    for cell in geometry.get_all_cells().values():          # one depletable material per pin
        if cell.fill is fuel or cell.fill is gd:
            cell.fill = cell.fill.clone()
            cell.fill.depletable = True
            cell.fill.volume = math.pi * E.fuel_r ** 2       # unit axial length
    materials = openmc.Materials(geometry.get_all_materials().values())
    materials.cross_sections = os.environ['OPENMC_CROSS_SECTIONS']
    settings = openmc.Settings()
    settings.batches, settings.inactive, settings.particles = BATCHES, INACTIVE, PARTICLES
    settings.seed = SEED_TRAJECTORY
    settings.output = {'summary': False}
    settings.source = openmc.IndependentSource(space=openmc.stats.Box(
        [-E.assembly_size / 2, -E.assembly_size / 2, 0], [E.assembly_size / 2, E.assembly_size / 2, 1]))
    return openmc.Model(geometry, materials, settings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', choices=['DPO', 'GRPO', 'GA', 'REF16', 'REF24'], required=True)
    ap.add_argument('--layouts', type=Path, default=None,
                    help='selected_layouts.json (default: $DATA_ROOT/depletion/selected_layouts.json, '
                         'then $DATA_ROOT/analysis/selected_layouts.json)')
    ap.add_argument('--output-root', type=Path, default=HERE / 'results')
    ap.add_argument('--smoke', action='store_true', help='2,000 particles, 0-0.01 MWd/kg; API check only')
    a = ap.parse_args()
    if not CHAIN.exists():
        sys.exit(f'{CHAIN} missing; run fetch_chain.py first')
    layouts_path = a.layouts or next((p for p in [DATA_ROOT / 'depletion/selected_layouts.json',
                                                  DATA_ROOT / 'analysis/selected_layouts.json'] if p.exists()), None)
    if layouts_path is None:
        sys.exit('selected_layouts.json not found; pass --layouts or run analysis/first_passage.py')
    layout = json.loads(layouts_path.read_text())[a.case]

    folder = (a.output_root / ('smoke' if a.smoke else '')) / a.case
    folder.mkdir(parents=True, exist_ok=True)
    os.chdir(folder)
    if Path('complete.json').exists():
        print('Already complete')
        return
    model = make_model(layout['grid'])
    if a.smoke:
        model.settings.particles, model.settings.batches, model.settings.inactive = 2000, 15, 5
    nodes = [0, .01] if a.smoke else BURNUPS
    Path('protocol.json').write_text(json.dumps({
        'case': a.case, 'grid': layout['grid'], 'smoke': a.smoke,
        'burnup_nodes_MWd_per_kg_initial_HM': nodes, 'power_density_W_per_g_initial_HM': POWER_DENSITY,
        'chain': str(CHAIN), 'integrator': 'CECM', 'pinwise_depletion': True,
        'particles_per_batch': model.settings.particles, 'batches': model.settings.batches,
        'inactive': model.settings.inactive,
        'residual_definition': 'endpoint k with remaining Gd isotopes removed at fixed other nuclide '
                               'densities minus unmodified endpoint k',
        'scope': '2D infinite lattice; unity crossing is not reactor cycle length; no thermal feedback; '
                 'exploratory, no time-step convergence established'}, indent=2))
    model.export_to_model_xml()

    previous = openmc.deplete.Results('depletion_results.h5') if Path('depletion_results.h5').exists() else None
    completed = len(previous) - 1 if previous else 0
    op = openmc.deplete.CoupledOperator(model, str(CHAIN), prev_results=previous)
    if completed < len(nodes) - 1:
        integrator = openmc.deplete.CECMIntegrator(op, np.diff(nodes)[completed:],
                                                   power_density=POWER_DENSITY, timestep_units='MWd/kg')
        integrator.integrate()
    results = openmc.deplete.Results('depletion_results.h5')
    times, keff = results.get_keff()
    assert len(keff) == len(nodes), (len(keff), len(nodes))
    with Path('trajectory.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['burnup_MWd_per_kg_initial_HM', 'days', 'k_infinity', 'k_sd'])
        w.writerows(zip(nodes, times / 86400, keff[:, 0], keff[:, 1]))

    # Endpoint composition with only the remaining Gd isotopes removed (no fresh-fuel substitution).
    model = openmc.Model.from_model_xml('model.xml')
    mats = results.export_to_materials(len(results) - 1, path='materials.xml')
    byid = {m.id: m for m in mats}
    removed = 0
    for m in mats:
        if m.depletable:
            names = [n for n in m.get_nuclide_atom_densities() if n.startswith('Gd')]
            if names:
                for n in names:
                    m.remove_nuclide(n)
                    removed += 1
                m.set_density('sum')          # keep exported absolute atom densities of the rest
    for cell in model.geometry.get_all_cells().values():
        if isinstance(cell.fill, openmc.Material):
            cell.fill = byid[cell.fill.id]
    model.materials = mats
    model.settings.seed = SEED_ENDPOINT
    endpoint = folder / 'endpoint_without_residual_gd'
    endpoint.mkdir(exist_ok=True)
    sp_path = model.run(cwd=endpoint, output=True)
    with openmc.StatePoint(sp_path) as sp:
        kg, ks = float(sp.keff.n), float(sp.keff.s)

    crossings = [float(nodes[i - 1] + (1 - keff[i - 1, 0]) * (nodes[i] - nodes[i - 1]) / (keff[i, 0] - keff[i - 1, 0]))
                 for i in range(1, len(nodes)) if keff[i - 1, 0] >= 1 and keff[i, 0] < 1]
    Path('complete.json').write_text(json.dumps({
        'smoke': a.smoke, 'case': a.case,
        'BOL_k': float(keff[0, 0]), 'endpoint_k': float(keff[-1, 0]), 'endpoint_k_sd': float(keff[-1, 1]),
        'last_downward_unity_crossing_MWd_per_kg_initial_HM': crossings[-1] if crossings else None,
        'all_downward_unity_crossings': crossings,
        'endpoint_without_residual_Gd_k': kg,
        'residual_Gd_penalty_delta_k': kg - float(keff[-1, 0]),
        'residual_penalty_sd_independent': float(np.hypot(ks, keff[-1, 1])),
        'removed_Gd_nuclide_entries': removed}, indent=2))
    print(f'{a.case}: done')


if __name__ == '__main__':
    main()
