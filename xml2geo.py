#!/usr/bin/env python3
"""Extract the ECAL barrel description from the k4geo compact files into the flat file read by ./sim and ntuple.py.
  python xml2geo.py /path/to/k4geo/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03 > geometry.txt
Lengths in mm, angles in rad, densities in g/cm3. Nothing about the calorimeter is hard-coded here."""
import sys, math, xml.etree.ElementTree as ET

d = sys.argv[1]
ECAL = ET.parse(f"{d}/ECalBarrel_thetamodulemerged.xml").getroot()

# 1. constants: evaluate the expressions of DectDimensions.xml and of the ECAL file
ns = {k: getattr(math, k) for k in ("sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "pi")}
ns.update(mm=1., cm=10., m=1000., um=1e-3, rad=1., mrad=1e-3, deg=math.pi / 180, degree=math.pi / 180, tesla=1., T=1.)
todo = [c.attrib for f in (ET.parse(f"{d}/DectDimensions.xml").getroot(), ECAL) for c in f.iter("constant")]
ev = lambda expr: float(eval(expr, {"__builtins__": {}}, ns))   # the files come from the k4geo repository
while todo:
    left = []
    for c in todo:
        try: ns[c["name"]] = ev(c["value"])
        except (NameError, SyntaxError, TypeError): left.append(c)   # depends on something not defined yet / not numeric
    if len(left) == len(todo): break
    todo = left

# 2. the barrel detector element
det = next(x for x in ECAL.iter("detector") if x.get("type", "").startswith("ECalBarrel_NobleLiquid_InclinedTrapezoids"))
cryo, calo = det.find("cryostat"), det.find("calorimeter")
passive, active, readout = calo.find("passive"), calo.find("active"), calo.find("readout")
A = lambda node, k: ev(node.get(k))
out = {k: A(calo.find("dimensions"), k) for k in ("rmin", "rmax", "dz")}
out.update({"cryo_" + k: A(cryo.find("dimensions"), k) for k in ("rmin1", "rmin2", "rmax1", "rmax2", "dz")})
out.update(angle=A(passive.find("rotation"), "angle"), t_active=A(active, "thickness"), t_readout=A(readout, "thickness"),
           t_inner=A(passive.find("inner"), "thickness"), t_inner_max=A(passive.find("innerMax"), "thickness"),
           t_glue=A(passive.find("glue"), "thickness"), t_outer=A(passive.find("outer"), "thickness"))
assert out["t_inner"] == out["t_inner_max"], "trapezoidal absorbers (Pb_thickness_max != Pb_thickness) are not implemented in sim.cc"
# number of planes: same formula as ECalBarrel_NobleLiquid_InclinedTrapezoids_o1_v03_geo.cpp
tp = out["t_inner"] + out["t_glue"] + out["t_outer"]
out["nplanes"] = round(math.pi / math.asin((tp + out["t_active"] + out["t_readout"]) / (2 * out["rmin"] * math.cos(out["angle"]))))
layers = [A(l, "thickness") for l in calo.find("layers") for _ in range(int(l.get("repeat", 1)))]

# 3. readout segmentation
seg = next(r for r in ECAL.iter("readout") if r.get("name") == det.get("readout")).find("segmentation")
assert ev(seg.get("nModules")) == out["nplanes"], "nModules of the segmentation != number of planes from the geometry"
out.update(theta_grid=A(seg, "grid_size_theta"), theta_offset=A(seg, "offset_theta"))

# 4. materials
mats = {"active": active.find("material").get("name"), "core0": passive.find("inner/material").get("name"),
        "core": passive.find("innerMax/material").get("name"), "glue": passive.find("glue/material").get("name"),
        "outer": passive.find("outer/material").get("name"), "readout": readout.find("material").get("name"),
        "cryo": cryo.find("material").get("name")}
defs = {m.get("name"): m for f in ("elements.xml", "materials.xml") for m in ET.parse(f"{d}/{f}").getroot().iter("material")}

for k, v in out.items(): print(k, repr(v))
print("layers", *layers)
print("merge_theta", seg.get("mergedCells_Theta"))
print("merge_module", seg.get("mergedModules"))
# choices for the test-beam module (not from the xml)
print("sector_half_deg 30   # half width of the simulated sector in azimuth; 180 = whole ring")
print("z_half 1500          # half length of the simulated sector along z [mm]")
print("use_cryo_back 0      # 0: virtual detector right behind the liquid, as in the paper; 1: behind the back cryostat wall (incl. solenoid)")
for role, name in mats.items(): print("mat_" + role, name)
for name in sorted(set(mats.values())):
    m = defs[name]
    assert m.find("D").get("unit", "g/cm3") == "g/cm3"
    comp = [("natoms", c) for c in m.findall("composite")] + [("fraction", c) for c in m.findall("fraction")]
    assert len({k for k, _ in comp}) == 1 and all(len(c.get("ref")) <= 2 for _, c in comp), f"material {name}: unsupported composition"
    print("material", name, m.find("D").get("value"), comp[0][0], *[x for _, c in comp for x in (c.get("ref"), c.get("n"))])
