"""Blender (Cycles) renders of the 17 x 17 PWR assembly for Figure 1.

Geometry follows claude_figure/revised_nature_3/render_assembly2.py (pitch units,
1 = 1.26 cm): rod OD 0.950 cm, guide thimble OD 1.20 / ID 1.06 cm, five
egg-crate spacer grids, top nozzle with hold-down springs, bottom nozzle with
legs. Active length shortened to 34 pitches so the assembly fits a panel.

Modes
  input    training lattice (pattern 1029, 16 Gd rods), full assembly
  output   selected DPO design (33 Gd rods), cut at mid-height so the pattern
           shows on the cut face; guide thimbles continue above the cut

Usage
  Blender -b --python render_blender.py -- --mode input --out assets/blender_input.png
  optional: --samples 256 --scale 1.0
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="input", choices=["input", "output"])
ap.add_argument("--out", required=True)
ap.add_argument("--samples", type=int, default=256)
ap.add_argument("--scale", type=float, default=1.0, help="resolution scale, 1.0 = 2000 x 3400 px")
args = ap.parse_args(argv)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D = json.load(open(ROOT / "claude_figure/revised_fable1/fig1_data.json"))
grid = D["example"]["grid"] if args.mode == "input" else D["dpo_best"]["grid"]
assert len(grid) == 289

# ---------------------------------------------------------------- geometry
N, HALF = 17, 8.5
R_ROD, R_GT_O, R_GT_I = 0.950 / 2 / 1.26, 1.20 / 2 / 1.26, 1.06 / 2 / 1.26
L = 34.0
Z_GRIDS = [2.0, 9.5, 17.0, 24.5, 31.5]
GRID_H, STRAP_T, OUTER_T = 1.4, 0.055, 0.11
Z_CUT = 20.5 if args.mode == "output" else None


def srgb(hexstr):
    """sRGB hex to linear RGB for Blender's colour sockets."""
    r, g, b = (int(hexstr[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (lin(r), lin(g), lin(b), 1.0)


def material(name, color, metallic, roughness, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        try:
            m.surface_render_method = "BLENDED"
        except Exception:
            pass
    return m


MAT = {
    "fuel": material("fuel", srgb("D9DCE0"), 0.85, 0.36),
    "gd": material("gd", srgb("0072B2"), 0.25, 0.42),
    "gt": material("gt", srgb("F2F3F4"), 0.70, 0.30),
    "ghost": material("ghost", srgb("C9CED3"), 0.30, 0.25, alpha=0.35),
    "strap": material("strap", srgb("8E949A"), 0.90, 0.42),
    "nozzle": material("nozzle", srgb("6B7076"), 0.85, 0.48),
    "spring": material("spring", srgb("5B6066"), 0.90, 0.40),
    "hole": material("hole", srgb("202326"), 0.10, 0.90),
}

# clean scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
scene = bpy.context.scene
coll = scene.collection


def add_obj(name, mesh, loc=(0, 0, 0)):
    o = bpy.data.objects.new(name, mesh)
    o.location = loc
    coll.objects.link(o)
    return o


def cyl_mesh(name, r, h, mat, n=48):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=n, radius1=r, radius2=r, depth=h)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    me.materials.append(mat)
    for p in me.polygons:
        p.use_smooth = True
    # keep the caps flat
    for p in me.polygons:
        if abs(p.normal.z) > 0.9:
            p.use_smooth = False
    return me


def tube_mesh(name, ri, ro, h, mat, n=48):
    bm = bmesh.new()
    zs = (-h / 2, h / 2)
    ring = {}
    for rad, tag in ((ro, "o"), (ri, "i")):
        for z in zs:
            ring[(tag, z)] = [bm.verts.new((rad * math.cos(2 * math.pi * k / n), rad * math.sin(2 * math.pi * k / n), z)) for k in range(n)]
    bm.verts.ensure_lookup_table()
    def quads(a, b):
        for k in range(n):
            bm.faces.new((a[k], a[(k + 1) % n], b[(k + 1) % n], b[k]))
    quads(ring[("o", zs[0])], ring[("o", zs[1])])      # outer wall
    quads(ring[("i", zs[1])], ring[("i", zs[0])])      # inner wall
    quads(ring[("o", zs[1])], ring[("i", zs[1])])      # top ring
    quads(ring[("i", zs[0])], ring[("o", zs[0])])      # bottom ring
    bm.normal_update()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    me.materials.append(mat)
    for p in me.polygons:
        p.use_smooth = abs(p.normal.z) < 0.9
    return me


def box(name, cx, cy, cz, lx, ly, lz, mat):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= lx; v.co.y *= ly; v.co.z *= lz
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    me.materials.append(mat)
    return add_obj(name, me, (cx, cy, cz))


# rods and guide thimbles (shared meshes, instanced objects)
rod_h = L if Z_CUT is None else Z_CUT
m_fuel = cyl_mesh("fuel_rod", R_ROD, rod_h, MAT["fuel"])
m_gd = cyl_mesh("gd_rod", R_ROD, rod_h, MAT["gd"])
gt_h = (L + 3.0) if Z_CUT is None else Z_CUT + 0.6
m_gt = tube_mesh("guide", R_GT_I, R_GT_O, gt_h, MAT["gt"])
m_ghost = tube_mesh("guide_ghost", R_GT_I, R_GT_O, 6.0, MAT["ghost"], n=32) if Z_CUT else None
for i in range(N):
    for j in range(N):
        ch = grid[i * N + j]
        cx, cy = j - HALF + 0.5, HALF - i - 0.5
        if ch == "c":
            add_obj(f"gt_{i}_{j}", m_gt, (cx, cy, -0.6 + gt_h / 2))
            if m_ghost is not None:
                add_obj(f"ghost_{i}_{j}", m_ghost, (cx, cy, Z_CUT + 3.0))
        else:
            add_obj(f"rod_{i}_{j}", m_gd if ch == "g" else m_fuel, (cx, cy, rod_h / 2))

# spacer grids (egg-crate straps)
for gi, zc in enumerate(Z_GRIDS):
    if Z_CUT is not None and zc > Z_CUT:
        continue
    for k in range(N + 1):
        t = OUTER_T if k in (0, N) else STRAP_T
        box(f"strapy_{gi}_{k}", 0, HALF - k, zc, 2 * HALF + OUTER_T, t, GRID_H, MAT["strap"])
        box(f"strapx_{gi}_{k}", k - HALF, 0, zc, t, 2 * HALF + OUTER_T, GRID_H, MAT["strap"])

# nozzles
plate_w = 2 * HALF + 0.5
box("bottom_plate", 0, 0, -0.95, plate_w, plate_w, 0.7, MAT["nozzle"])
for sx in (-1, 1):
    for sy in (-1, 1):
        box(f"leg_{sx}_{sy}", sx * (HALF - 0.3), sy * (HALF - 0.3), -2.2, 1.4, 1.4, 1.8, MAT["nozzle"])
if Z_CUT is None:
    box("top_plate", 0, 0, L + 0.75, plate_w, plate_w, 0.7, MAT["nozzle"])
    for sx in (-1, 1):
        box(f"wallx_{sx}", sx * (plate_w / 2 - 0.12), 0, L + 1.6, 0.24, plate_w, 1.0, MAT["nozzle"])
        box(f"wally_{sx}", 0, sx * (plate_w / 2 - 0.12), L + 1.6, plate_w, 0.24, 1.0, MAT["nozzle"])
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(f"spring_{sx}_{sy}", sx * 2.6, sy * (HALF - 1.1), L + 2.4, 4.0, 0.9, 0.35, MAT["spring"])
    m_hole = cyl_mesh("hole", 0.9, 0.06, MAT["hole"], n=32)
    for a in range(4):
        for b in range(4):
            add_obj(f"hole_{a}_{b}", m_hole, ((a - 1.5) * 3.4, (b - 1.5) * 3.4, L + 1.13))

# ---------------------------------------------------------------- camera and lights
target = bpy.data.objects.new("target", None)
coll.objects.link(target)
target.location = (0, 0, L / 2 + 1.0) if Z_CUT is None else (0, 0, Z_CUT / 2 + 2.0)
cam_data = bpy.data.cameras.new("cam")
cam_data.sensor_fit = "VERTICAL"
cam_data.angle_y = math.radians(32)   # vertical field of view, set after the fit mode
cam = bpy.data.objects.new("cam", cam_data)
cam.location = (50, -66, 78) if Z_CUT is None else (44, -58, 62)
coll.objects.link(cam)
tr = cam.constraints.new("TRACK_TO")
tr.target = target; tr.track_axis = "TRACK_NEGATIVE_Z"; tr.up_axis = "UP_Y"
scene.camera = cam


def area_light(name, loc, energy, size, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy; ld.size = size; ld.color = color
    lo = bpy.data.objects.new(name, ld)
    lo.location = loc
    coll.objects.link(lo)
    c = lo.constraints.new("TRACK_TO")
    c.target = target; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
    return lo


area_light("key", (-45, -75, 120), 2.2e5, 45)
area_light("fill", (95, -40, 55), 7.0e4, 60, (0.97, 0.98, 1.0))
area_light("rim", (25, 90, 70), 9.0e4, 30)
scene.world = bpy.data.worlds.new("world")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.85, 0.87, 0.90, 1.0)
bg.inputs[1].default_value = 0.35

# ---------------------------------------------------------------- render settings
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "METAL"
prefs.get_devices()
for d in prefs.devices:
    d.use = (d.type == "METAL")
scene.cycles.device = "GPU"
scene.cycles.samples = args.samples
scene.cycles.use_denoising = True
scene.cycles.use_adaptive_sampling = True
scene.render.film_transparent = True
scene.render.resolution_x, scene.render.resolution_y = 2000, 3400
scene.render.resolution_percentage = int(100 * args.scale)
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "16"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.render.filepath = str(Path(args.out).resolve())
bpy.ops.render.render(write_still=True)
print("wrote", scene.render.filepath)
