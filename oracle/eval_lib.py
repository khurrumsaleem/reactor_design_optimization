"""OpenMC evaluation utilities for the informed random-search oracle.

Reproduces the geometry, materials, settings, tallies, and composite fitness
of the GA baselines (ga/ga_baseline.py) exactly, so oracle results are
directly comparable with the GA and alignment campaigns.
"""
import os

os.environ.setdefault(
    'OPENMC_CROSS_SECTIONS',
    os.path.expanduser('~/openmc_data/endfb-vii.1-hdf5/cross_sections.xml'),
)

import shutil
import tempfile

import numpy as np
import openmc

pitch, fuel_r, clad_id, clad_od = 1.26, 0.4096, 0.835, 0.950
assembly_size = 17 * pitch
TARGET_K_EFF = 1.05

GT_COORDS = [
    (2, 5), (2, 8), (2, 11),
    (3, 3), (3, 13),
    (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
    (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
    (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
    (13, 3), (13, 13),
    (14, 5), (14, 8), (14, 11),
]
GT_INDICES = [r * 17 + c for r, c in GT_COORDS]
VALID_INDICES = [i for i in range(289) if i not in GT_INDICES]


def create_materials():
    fuel = openmc.Material(name='UO2 3.1wt%')
    fuel.add_nuclide('U235', 0.031)
    fuel.add_nuclide('U238', 0.969)
    fuel.add_nuclide('O16', 2.0)
    fuel.set_density('g/cm3', 10.29769)

    gd = openmc.Material(name='Gd Poison 8wt%')
    gd.add_nuclide('U235', 0.031 * 0.92)
    gd.add_nuclide('U238', 0.969 * 0.92)
    gd.add_nuclide('O16', 2.0 * 0.92 + 3.0 * 0.08)
    gd.add_nuclide('Gd155', 0.08 * 0.148)
    gd.add_nuclide('Gd156', 0.08 * 0.199)
    gd.add_nuclide('Gd157', 0.08 * 0.156)
    gd.add_nuclide('Gd158', 0.08 * 0.249)
    gd.add_nuclide('Gd160', 0.08 * 0.218)
    gd.set_density('g/cm3', 10.5)

    zirc = openmc.Material(name='Zircaloy-4')
    for el, frac in [('Sn', 0.014), ('Fe', 0.0121), ('Cr', 0.0107), ('Ni', 0.0050), ('Zr', 0.9582)]:
        zirc.add_element(el, frac)
    zirc.set_density('g/cm3', 6.55)

    water = openmc.Material(name='PWR Water')
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.add_s_alpha_beta('c_H_in_H2O')
    water.set_density('g/cm3', 0.701)

    materials = openmc.Materials([fuel, gd, zirc, water])
    materials.export_to_xml()
    return fuel, gd, zirc, water


def create_assembly(grid, fuel, gd, zirc, water):
    def get_univ(rod_type):
        fuel_cyl = openmc.ZCylinder(r=fuel_r)
        clad_inner = openmc.ZCylinder(r=clad_id / 2)
        clad_outer = openmc.ZCylinder(r=clad_od / 2)
        box = openmc.model.RectangularPrism(width=pitch, height=pitch, boundary_type='reflective')
        fill = {0: fuel, 1: gd, 2: water}[rod_type]
        cells = [openmc.Cell(fill=fill, region=-fuel_cyl),
                 openmc.Cell(fill=None, region=+fuel_cyl & -clad_inner),
                 openmc.Cell(fill=zirc, region=+clad_inner & -clad_outer),
                 openmc.Cell(fill=water, region=-box & +clad_outer)]
        return openmc.Universe(cells=cells)

    lattice = openmc.RectLattice()
    lattice.pitch = (pitch, pitch)
    lattice.lower_left = (-assembly_size / 2, -assembly_size / 2)
    lattice.dimension = (17, 17)
    lattice.universes = np.array([[get_univ(grid[i, j]) for j in range(17)] for i in range(17)])
    outer_box = openmc.model.RectangularPrism(assembly_size, assembly_size, boundary_type='reflective')
    return openmc.Geometry([openmc.Cell(fill=lattice, region=-outer_box)])


def evaluate_grid(grid_str, sim_seed=None):
    """Evaluate one 289-character layout. Returns (fitness, k_eff, fq, fdh, g_count, grid_str)."""
    g_count = grid_str.count('g')
    work_dir = tempfile.mkdtemp(prefix='oracle_run_', dir=tempfile.gettempdir())
    original_dir = os.getcwd()

    try:
        os.chdir(work_dir)
        fuel, gd, zirc, water = create_materials()
        grid = np.zeros((17, 17), dtype=int)
        for i, char in enumerate(grid_str):
            if char == 'g':
                grid[divmod(i, 17)] = 1
            elif char == 'c':
                grid[divmod(i, 17)] = 2

        create_assembly(grid, fuel, gd, zirc, water).export_to_xml()
        settings = openmc.Settings()
        settings.batches = 30
        settings.inactive = 10
        settings.particles = 20000
        settings.output = {'summary': False}
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Box([-assembly_size / 2, -assembly_size / 2, 0],
                                   [assembly_size / 2, assembly_size / 2, 1]))
        if sim_seed is not None:
            settings.seed = int(sim_seed)
        settings.export_to_xml()

        tallies = openmc.Tallies()
        mesh = openmc.RegularMesh()
        mesh.dimension = (17, 17, 1)
        mesh.lower_left = (-assembly_size / 2, -assembly_size / 2, 0)
        mesh.upper_right = (assembly_size / 2, assembly_size / 2, 1)
        pt = openmc.Tally(name='power')
        pt.filters = [openmc.MeshFilter(mesh)]
        pt.scores = ['fission']
        tallies.append(pt)
        tallies.export_to_xml()

        openmc.run(output=False)
        with openmc.StatePoint('statepoint.30.h5') as sp:
            k_eff = sp.keff.nominal_value
            power = sp.get_tally(name='power').mean.ravel()
            fq = power.max() / power.mean() if power.mean() > 0 else 99.9
            ch = power.reshape(17, 17).mean(axis=0)
            fdh = ch.max() / ch.mean() if ch.mean() > 0 else 99.9

        fitness = 0.6 * fq + 0.4 * fdh + 100.0 * abs(k_eff - TARGET_K_EFF)
        return (fitness, k_eff, fq, fdh, g_count, grid_str)
    except Exception:
        return (99.9, 0.0, 99.9, 99.9, g_count, grid_str)
    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)


def random_layout(rng, gd_min=20, gd_max=41):
    """Uniform random layout: Gd count uniform on [gd_min, gd_max), positions uniform."""
    ind = ['f'] * 289
    for i in GT_INDICES:
        ind[i] = 'c'
    num_g = int(rng.integers(gd_min, gd_max))
    for pos in rng.choice(VALID_INDICES, num_g, replace=False):
        ind[pos] = 'g'
    return ''.join(ind)
