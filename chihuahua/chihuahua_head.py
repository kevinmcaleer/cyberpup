"""
CyberPup - robot chihuahua head built from flat, slot-together plates.

Same construction idea as the robot cat: every part is a flat 3 mm profile
(3D print lying flat, or laser cut) and the 3D head is built up by slotting
the profiles into each other with tabs and windows.

The HC-SR04 ultrasonic sensor provides the (suitably bulging) chihuahua eyes.

Run it:
  * Blender GUI:  Scripting tab -> open this file -> Run Script
                  (builds the assembly + print layout in the scene)
  * Headless:     blender -b -P chihuahua_head.py -- --out build --render
  * pip bpy:      python chihuahua_head.py --out build --render

Outputs (when --out is given): STL per part (print orientation), SVG + DXF
outlines per part (laser cutting / import as a Fusion 360 sketch), the .blend
file, and optional preview renders.

All dimensions are millimetres. Change the parameters below and re-run.
"""

import argparse
import math
import os
import sys

import bpy  # noqa: I001  (bpy must be imported before bmesh/mathutils)
import bmesh
from mathutils import Matrix, Vector

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
T = 3.0             # plate thickness
CLR = 0.15          # clearance per side for every window / slot
TAB = T - 0.2       # how far a tab goes into a window (stays just under flush)

# HC-SR04 ultrasonic sensor (the eyes)
SENSOR_W = 45.0     # PCB width
SENSOR_H = 20.0     # PCB height
SENSOR_PCB = 1.6    # PCB thickness
EYE_D = 16.0        # transducer can diameter
EYE_SPACING = 26.0  # centre-to-centre distance of the cans
EYE_HOLE = EYE_D + 0.6
SENSOR_ENGAGE = 1.0  # how far the PCB edges sit into the side plate pockets

# Head proportions
SKULL_R = 31.0      # apple-dome skull radius (front view)
SKULL_CZ = 6.0      # skull centre height (eyes are at z = 0)
CHEEK_RX, CHEEK_RZ, CHEEK_CZ = 25.0, 21.0, -6.0   # cheek / chin ellipse
EAR_LEN = 44.0      # chihuahua ears: big!
EAR_W = 40.0        # ear width at the base
EAR_TILT = 56.0     # ear angle from vertical (degrees)
EAR_POS = 50.0      # where the ear sits on the skull (degrees from top)

# Plate positions (world: x = right, y = forward, z = up, eyes at origin)
SIDE_X = SENSOR_W / 2 + T / 2 - SENSOR_ENGAGE  # side plates grip the PCB edges
MUZZLE_Z = -15.0    # bottom of the muzzle plate
BASE_Z = -22.0      # bottom of the base / neck plate
CROWN_Z = 18.0      # bottom of the crown plate
NOSE_Y = 27.6       # front face of the nose (flush with the snout tip)

# Neck servo horn (SG90 / MG90 style) on the base plate
HORN_CENTRE = (0.0, -19.0)
HORN_HUB_D = 7.0
HORN_SLOT = (6.0, 16.0, 1.8)   # slot from r=6 to r=16, 1.8 wide, both sides

COLOURS = {
    "face": (1.0, 0.78, 0.05), "side": (1.0, 0.78, 0.05),
    "snout": (0.95, 0.95, 0.92), "muzzle": (0.95, 0.95, 0.92),
    "crown": (1.0, 0.78, 0.05), "base": (0.95, 0.95, 0.92),
    "nose": (0.03, 0.03, 0.03), "eye_ring": (0.03, 0.03, 0.03),
}

# ---------------------------------------------------------------------------
# 2D helpers (polygons are lists of (x, y) tuples)
# ---------------------------------------------------------------------------


def ellipse(cx, cy, rx, ry, n=96):
    return [(cx + rx * math.cos(2 * math.pi * i / n),
             cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def circle(cx, cy, r, n=96):
    return ellipse(cx, cy, r, r, n)


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def window(x0, y0, x1, y1):
    """Rectangle grown by the clearance, for a tab of size (x0,y0)-(x1,y1)."""
    return rect(x0 - CLR, y0 - CLR, x1 + CLR, y1 + CLR)


def mirror_x(poly):
    return [(-x, y) for x, y in reversed(poly)]


def clip(poly, a, b, c):
    """Keep the part of poly where a*x + b*y <= c (Sutherland-Hodgman)."""
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        dp = a * p[0] + b * p[1] - c
        dq = a * q[0] + b * q[1] - c
        if dp <= 0:
            out.append(p)
        if dp * dq < 0:
            t = dp / (dp - dq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def ear(length, width, t0=0.0, base_ext=0.0, n=24, p_out=0.55, p_in=0.7):
    """Right ear in ear coords: s = across (+ outer side), t = along."""
    pts = [(-width / 2, t0 - base_ext), (width / 2, t0 - base_ext)]
    for i in range(n + 1):
        f = min(i / n, 0.97)
        pts.append((width / 2 * (1 - f) ** p_out, t0 + length * f))
    for i in range(n, -1, -1):
        f = min(i / n, 0.97)
        pts.append((-width / 2 * (1 - f) ** p_in, t0 + length * f))
    return pts


def place_ear(pts):
    b, a = math.radians(EAR_POS), math.radians(EAR_TILT)
    r = SKULL_R - 7.0
    bx, bz = r * math.sin(b), SKULL_CZ + r * math.cos(b)
    d = (math.sin(a), math.cos(a))       # along the ear
    s = (math.cos(a), -math.sin(a))      # across, towards the outside
    return [(bx + u * s[0] + v * d[0], bz + u * s[1] + v * d[1]) for u, v in pts]


# ---------------------------------------------------------------------------
# Part definitions: each part is a flat plate, local (u, v) plane, thickness
# along local w.  `adds` are unioned, `subs` are cut out.
# ---------------------------------------------------------------------------


class Part:
    def __init__(self, name, qty=1, thickness=T):
        self.name, self.qty, self.thickness = name, qty, thickness
        self.adds, self.subs = [], []
        self.placements = []   # world matrices for the assembly

    def add(self, *polys):
        self.adds.extend(polys)
        return self

    def sub(self, *polys):
        self.subs.extend(polys)
        return self


def m_face(y_front):
    """Plate in the XZ plane, occupying y in [y_front - t, y_front]."""
    return Matrix.Translation((0, y_front, 0)) @ Matrix.Rotation(math.radians(90), 4, 'X')


def m_side(x_centre, t=T):
    """Plate in the YZ plane centred on x_centre."""
    m = Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    return Matrix.Translation((x_centre - t / 2, 0, 0)) @ m


def m_flat(z_bottom):
    """Plate in the XY plane with its bottom at z_bottom."""
    return Matrix.Translation((0, 0, z_bottom))


def both_sides(x0, y0, x1, y1, fn):
    return [fn(x0, y0, x1, y1), fn(-x1, y0, -x0, y1)]


def define_parts():
    parts = []
    ex = EYE_SPACING / 2

    # --- Face plate (XZ): skull, cheeks, ears, eye holes, all the windows ---
    face = Part("face")
    right_ear = place_ear(ear(EAR_LEN, EAR_W, base_ext=8))
    face.add(circle(0, SKULL_CZ, SKULL_R),
             ellipse(0, CHEEK_CZ, CHEEK_RX, CHEEK_RZ),
             right_ear, mirror_x(right_ear))
    inner = place_ear(ear(EAR_LEN * 0.6, EAR_W * 0.5, t0=11))
    face.sub(inner, mirror_x(inner),
             circle(ex, 0, EYE_HOLE / 2), circle(-ex, 0, EYE_HOLE / 2))
    face.sub(window(-T / 2, -6, T / 2, -1), window(-T / 2, 3, T / 2, 8))      # snout
    face.sub(*both_sides(6, MUZZLE_Z, 13, MUZZLE_Z + T, window))               # muzzle
    face.sub(*both_sides(7, BASE_Z, 11, BASE_Z + T, window))                   # base
    face.sub(*both_sides(SIDE_X - T / 2, 11, SIDE_X + T / 2, 21, window))      # sides
    face.placements = [m_face(0.0)]
    parts.append(face)

    # --- Snout fin (YZ at x=0): nose bridge + muzzle profile ---------------
    snout = Part("snout")
    top = MUZZLE_Z + T
    profile = [(0, top), (27, top), (27.6, top + 1), (27.6, -5.5), (27, -4.5),
               (20, -4.5), (14, -3.2), (10.5, -1.5), (8, 1), (6, 4.5),
               (4, 8), (2, 10.5), (0, 11.5)]
    snout.add(profile)
    snout.add(rect(-TAB, -6, 0.5, -1), rect(-TAB, 3, 0.5, 8))                 # into face
    snout.add(rect(4, top - TAB, 9, top + 0.5), rect(13, top - TAB, 17, top + 0.5))  # into muzzle
    snout.placements = [m_side(0.0)]
    parts.append(snout)

    # --- Muzzle plate (XY): top view of the muzzle ---------------------------
    muzzle = Part("muzzle")
    muzzle.add(clip(ellipse(0, 0, 17, 21), 0, -1, 0))
    muzzle.add(*both_sides(6, -TAB, 13, 0.5, rect))                           # into face
    muzzle.sub(window(-T / 2, 4, T / 2, 9), window(-T / 2, 13, T / 2, 17))    # snout tabs
    muzzle.placements = [m_flat(MUZZLE_Z)]
    parts.append(muzzle)

    # --- Side plates (YZ at x=+-SIDE_X): skull depth, grip the sensor PCB ---
    side = Part("side", qty=2)
    shape = ellipse(-14, 2, 26, 25)
    shape = clip(shape, 1, 0, -T)                  # behind the face plate
    shape = clip(shape, 0, -1, -(BASE_Z + T))      # sits on the base plate
    side.add(shape)
    side.add(rect(-T - 0.5, 11, -T + TAB, 21))                                # into face
    side.add(rect(-10, BASE_Z + T - TAB, -5, BASE_Z + T + 0.5),
             rect(-22, BASE_Z + T - TAB, -17, BASE_Z + T + 0.5))              # into base
    side.sub(rect(-T - SENSOR_PCB - 0.3, -SENSOR_H / 2 - 0.3, 0, SENSOR_H / 2 + 0.3))  # PCB pocket
    side.sub(window(-20, CROWN_Z, -10, CROWN_Z + T))                          # crown tabs
    side.placements = [m_side(SIDE_X), m_side(-SIDE_X)]
    parts.append(side)

    # --- Crown plate (XY): ties the side plates together over the skull ----
    crown = Part("crown")
    half = SIDE_X - T / 2 - 0.2
    crown.add(rect(-half, -20, half, -6), ellipse(0, -20, half, 8))
    crown.add(*both_sides(half - 0.5, -20, SIDE_X - T / 2 + TAB, -10, rect))
    crown.sub(circle(0, -16, 6))                                              # cable hole
    crown.placements = [m_flat(CROWN_Z)]
    parts.append(crown)

    # --- Base / neck plate (XY): servo horn mount ----------------------------
    base = Part("base")
    hx, hy = HORN_CENTRE
    base.add(rect(-SIDE_X - 4, -24, SIDE_X + 4, -T), ellipse(0, -24, SIDE_X + 4, 8))
    base.add(*both_sides(7, -T - 0.5, 11, -T + TAB, rect))                     # into face
    for y0, y1 in ((-10, -5), (-22, -17)):
        base.sub(*both_sides(SIDE_X - T / 2, y0, SIDE_X + T / 2, y1, window))
    base.sub(rect(-6.5, -12, 6.5, 1))                                         # sensor pins / cable
    r0, r1, w = HORN_SLOT
    base.sub(circle(hx, hy, HORN_HUB_D / 2),
             rect(hx + r0, hy - w / 2, hx + r1, hy + w / 2),
             rect(hx - r1, hy - w / 2, hx - r0, hy + w / 2))
    base.placements = [m_flat(BASE_Z)]
    parts.append(base)

    # --- Nose (XZ): slides onto the tip of the snout fin ---------------------
    nose = Part("nose")
    nose.add(clip(ellipse(0, -5.5, 8.5, 9), 0, 1, -2.2))              # rounded dog nose
    nose.sub(window(-T / 2, top, T / 2, -4.5))
    nose.placements = [m_face(NOSE_Y)]
    parts.append(nose)

    # --- Eye rings (XZ): bezels around the transducers ------------------------
    ring = Part("eye_ring", qty=2, thickness=2.0)
    ring.add(circle(0, 0, 11))
    ring.sub(circle(0, 0, EYE_HOLE / 2))
    ring.placements = [Matrix.Translation((x, 2.0, 0)) @ m_face(0.0) for x in (ex, -ex)]
    parts.append(ring)
    return parts


# ---------------------------------------------------------------------------
# Blender geometry
# ---------------------------------------------------------------------------


def signed_area(poly):
    return 0.5 * sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(poly, poly[1:] + poly[:1]))


def prism(name, poly, z0, z1, coll):
    pts = []
    for p in poly:
        if not pts or (abs(p[0] - pts[-1][0]) + abs(p[1] - pts[-1][1])) > 1e-6:
            pts.append(p)
    if signed_area(pts) < 0:
        pts.reverse()
    n = len(pts)
    verts = [(x, y, z0) for x, y in pts] + [(x, y, z1) for x, y in pts]
    faces = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def build_part_mesh(part, scene):
    tmp_add = bpy.data.collections.new(part.name + "_add")
    tmp_sub = bpy.data.collections.new(part.name + "_sub")
    for c in (tmp_add, tmp_sub):
        scene.collection.children.link(c)
    t = part.thickness
    base = prism(part.name + "_base", part.adds[0], 0, t, scene.collection)
    for i, p in enumerate(part.adds[1:]):
        prism(f"{part.name}_a{i}", p, 0, t, tmp_add)
    for i, p in enumerate(part.subs):
        prism(f"{part.name}_s{i}", p, -1, t + 1, tmp_sub)
    for coll, op in ((tmp_add, 'UNION'), (tmp_sub, 'DIFFERENCE')):
        if coll.objects:
            m = base.modifiers.new(op, 'BOOLEAN')
            m.operation, m.operand_type, m.collection, m.solver = op, 'COLLECTION', coll, 'EXACT'
    dg = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(base.evaluated_get(dg))
    mesh.name = part.name
    for coll in (tmp_add, tmp_sub):
        for ob in list(coll.objects):
            bpy.data.objects.remove(ob)
        bpy.data.collections.remove(coll)
    bpy.data.objects.remove(base)
    return mesh


def material(name, rgb):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1)
    bsdf.inputs["Roughness"].default_value = 0.55
    mat.diffuse_color = (*rgb, 1)
    return mat


def new_collection(scene, name):
    old = bpy.data.collections.get(name)
    if old:
        for ob in list(old.objects):
            bpy.data.objects.remove(ob)
        bpy.data.collections.remove(old)
    c = bpy.data.collections.new(name)
    scene.collection.children.link(c)
    return c


def sensor_dummy(coll):
    """HC-SR04 stand-in for the assembly (not exported)."""
    pcb = material("pcb", (0.05, 0.18, 0.55))
    metal = material("metal", (0.8, 0.8, 0.82))
    obs = []
    bpy.ops.mesh.primitive_cube_add(size=1)
    b = bpy.context.active_object
    b.scale = (SENSOR_W, SENSOR_PCB, SENSOR_H)
    b.location = (0, -T - SENSOR_PCB / 2, 0)
    b.data.materials.append(pcb)
    obs.append(b)
    for x in (EYE_SPACING / 2, -EYE_SPACING / 2):
        bpy.ops.mesh.primitive_cylinder_add(radius=EYE_D / 2, depth=12, vertices=48,
                                            rotation=(math.radians(90), 0, 0))
        c = bpy.context.active_object
        c.location = (x, -T + 6, 0)
        c.data.materials.append(metal)
        obs.append(c)
    for ob in obs:
        for uc in ob.users_collection:
            uc.objects.unlink(ob)
        coll.objects.link(ob)
        ob.name = "HC-SR04"
    return obs


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def mesh_stats(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    manifold = all(e.is_manifold for e in bm.edges)
    parent = list(range(len(bm.verts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for e in bm.edges:
        a, b = find(e.verts[0].index), find(e.verts[1].index)
        parent[a] = b
    pieces = len({find(i) for i in range(len(bm.verts))})
    vol = bm.calc_volume()
    bm.free()
    return manifold, pieces, vol


def interference(objs, scene):
    """Volume shared by every pair of assembled parts (should all be ~0)."""
    hits = []
    dg = bpy.context.evaluated_depsgraph_get()
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            ba = [a.matrix_world @ Vector(c) for c in a.bound_box]
            bb = [b.matrix_world @ Vector(c) for c in b.bound_box]
            if any(min(p[k] for p in ba) > max(p[k] for p in bb) or
                   min(p[k] for p in bb) > max(p[k] for p in ba) for k in range(3)):
                continue
            tmp = bpy.data.objects.new("tmp", a.data.copy())
            tmp.matrix_world = a.matrix_world
            scene.collection.objects.link(tmp)
            m = tmp.modifiers.new("x", 'BOOLEAN')
            m.operation, m.object, m.solver = 'INTERSECT', b, 'EXACT'
            dg = bpy.context.evaluated_depsgraph_get()
            me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
            bm = bmesh.new()
            bm.from_mesh(me)
            vol = abs(bm.calc_volume())
            bm.free()
            if vol > 0.01:
                hits.append((a.name, b.name, vol))
            bpy.data.objects.remove(tmp)
            bpy.data.meshes.remove(me)
    return hits


# ---------------------------------------------------------------------------
# 2D export (SVG / DXF) from the top face of each flat part
# ---------------------------------------------------------------------------


def outline_loops(mesh, t):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    top = {f for f in bm.faces if f.normal.z > 0.99 and abs(f.calc_center_median().z - t) < 1e-4}
    edges = [e for e in bm.edges if sum(f in top for f in e.link_faces) == 1]
    nxt = {}
    for e in edges:
        a, b = e.verts
        nxt.setdefault(a.index, []).append(b.index)
        nxt.setdefault(b.index, []).append(a.index)
    co = {v.index: (v.co.x, v.co.y) for v in bm.verts}
    bm.free()
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, prev, cur = [], None, start
        while cur not in seen:
            seen.add(cur)
            loop.append(co[cur])
            cands = [n for n in nxt[cur] if n != prev and n not in seen]
            if not cands:
                break
            prev, cur = cur, cands[0]
        if len(loop) > 2:
            loops.append(loop)
    return loops


def write_svg(path, loops):
    xs = [p[0] for l in loops for p in l]
    ys = [p[1] for l in loops for p in l]
    x0, y1 = min(xs) - 2, max(ys) + 2
    w, h = max(xs) - x0 + 2, y1 - min(ys) + 2
    d = " ".join("M " + " L ".join(f"{x - x0:.3f},{y1 - y:.3f}" for x, y in l) + " Z" for l in loops)
    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.2f}mm" height="{h:.2f}mm" '
                f'viewBox="0 0 {w:.3f} {h:.3f}">\n'
                f'<path d="{d}" fill="none" stroke="#ff0000" stroke-width="0.1" fill-rule="evenodd"/>\n'
                f'</svg>\n')


def write_dxf(path, loops):
    out = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009",
           "9", "$INSUNITS", "70", "4", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    for l in loops:
        out += ["0", "POLYLINE", "8", "0", "66", "1", "70", "1"]
        for x, y in l:
            out += ["0", "VERTEX", "8", "0", "10", f"{x:.4f}", "20", f"{y:.4f}"]
        out += ["0", "SEQEND"]
    out += ["0", "ENDSEC", "0", "EOF"]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def setup_render(scene):
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 1400, 1050
    scene.render.film_transparent = False
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.92, 0.95, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
    ld = bpy.data.lights.new("key", 'AREA')
    ld.energy, ld.size = 3.0e6, 150
    key = bpy.data.objects.new("key", ld)
    key.location = (150, 200, 250)
    look_at(key, (0, 0, 0))
    scene.collection.objects.link(key)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.clip_end = 5000
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def render_view(scene, cam, path, loc, target, lens=50, ortho=None):
    cam.location = loc
    look_at(cam, target)
    if ortho:
        cam.data.type, cam.data.ortho_scale = 'ORTHO', ortho
    else:
        cam.data.type, cam.data.lens = 'PERSP', lens
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="export folder (STL/SVG/DXF/.blend)")
    ap.add_argument("--render", action="store_true", help="also render preview images")
    args, _ = ap.parse_known_args(argv)

    scene = bpy.context.scene
    if bpy.app.background:
        for ob in list(bpy.data.objects):
            bpy.data.objects.remove(ob)
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = 'MILLIMETERS'

    assembly = new_collection(scene, "Chihuahua assembly")
    layout = new_collection(scene, "Chihuahua print layout")
    parts = define_parts()

    meshes, assembled = {}, []
    print("\nPart            qty  manifold  pieces  volume(mm3)")
    for part in parts:
        mesh = build_part_mesh(part, scene)
        mesh.materials.append(material(part.name, COLOURS[part.name]))
        meshes[part.name] = mesh
        manifold, pieces, vol = mesh_stats(mesh)
        print(f"{part.name:<15} {part.qty:>3}  {str(manifold):<8}  {pieces:>6}  {vol:10.0f}")
        for i, mw in enumerate(part.placements):
            ob = bpy.data.objects.new(f"{part.name}{'_' + 'LR'[i] if len(part.placements) > 1 else ''}", mesh)
            ob.matrix_world = mw
            assembly.objects.link(ob)
            assembled.append(ob)

    sensor = sensor_dummy(assembly)

    # Print layout: simple shelf packing on a 200 mm wide bed, 5 mm gaps
    x = y = row_h = 0.0
    bed = (0.0, 0.0)
    for part in parts:
        mesh = meshes[part.name]
        xs = [v.co.x for v in mesh.vertices]
        ys = [v.co.y for v in mesh.vertices]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        for _ in range(part.qty):
            if x + w > 200:
                x, y, row_h = 0.0, y + row_h + 5, 0.0
            ob = bpy.data.objects.new(part.name + "_print", mesh)
            ob.location = (x - min(xs) + 150, y - min(ys) - 100, -40)
            layout.objects.link(ob)
            bed = (max(bed[0], x + w), max(bed[1], y + h))
            x += w + 5
            row_h = max(row_h, h)

    hits = interference(assembled + sensor, scene)
    print("\nInterference check:", "OK - no parts overlap" if not hits else "")
    for a, b, v in hits:
        print(f"  {a} <-> {b}: {v:.2f} mm3")

    if args.out:
        out = os.path.abspath(args.out)
        for sub in ("stl", "svg", "dxf", "renders"):
            os.makedirs(os.path.join(out, sub), exist_ok=True)
        for part in parts:
            mesh = meshes[part.name]
            ob = bpy.data.objects.new(part.name + "_export", mesh)
            scene.collection.objects.link(ob)
            for o in bpy.context.view_layer.objects:
                o.select_set(False)
            ob.select_set(True)
            bpy.context.view_layer.objects.active = ob
            name = f"{part.name}_x{part.qty}"
            bpy.ops.wm.stl_export(filepath=os.path.join(out, "stl", name + ".stl"),
                                  export_selected_objects=True, apply_modifiers=True)
            loops = outline_loops(mesh, part.thickness)
            write_svg(os.path.join(out, "svg", name + ".svg"), loops)
            write_dxf(os.path.join(out, "dxf", name + ".dxf"), loops)
            bpy.data.objects.remove(ob)
        if args.render:
            cam = setup_render(scene)
            layout.hide_render = True
            r = os.path.join(out, "renders")
            render_view(scene, cam, os.path.join(r, "hero.png"), (150, 190, 70), (0, -5, 12), 55)
            render_view(scene, cam, os.path.join(r, "front.png"), (0, 400, 15), (0, 0, 15), ortho=135)
            render_view(scene, cam, os.path.join(r, "side.png"), (400, 0, 15), (0, 0, 15), ortho=135)
            render_view(scene, cam, os.path.join(r, "back.png"), (-130, -170, 90), (0, -8, 5), 55)
            layout.hide_render, assembly.hide_render = False, True
            cx, cy = 150 + bed[0] / 2, -100 + bed[1] / 2
            render_view(scene, cam, os.path.join(r, "print_layout.png"), (cx, cy, 400),
                        (cx, cy, -40), ortho=max(bed[0], bed[1] * 4 / 3) + 20)
            assembly.hide_render = False
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "chihuahua_head.blend"))
        print("\nExported to", out)


if __name__ == "__main__":
    main()
