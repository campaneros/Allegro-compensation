#!/usr/bin/env python3
"""ddsim edm4hep file(s) -> flat 'events' tree.   python ntuple.py out.root sim_pi-_*.root [--thr 0.003]"""
import argparse, numpy as np, awkward as ak, uproot

# per-layer sampling fractions, copied from FCC-config .../ALLEGRO_o1_v03/run_digi_reco.py (ddsim values)
SF = np.array([0.3790943904011486, 0.1355600584387894, 0.14628210607758893, 0.15274136994224854,
               0.15817255837886351, 0.16290355087527258, 0.1674201708055751, 0.1715846423182708,
               0.17558662106635545, 0.18002243792463576, 0.18288329976007917])
R0 = 2100 + 49 + 13.8 + 10  # EMBarrel_rmin [mm]
EC = "ECalBarrelModuleThetaMerged"

def field(cid, off, n):  # cellID = system:4,cryo:1,type:3,subtype:3,layer:8,module:11,theta:10
    return (cid >> np.uint64(off)) & np.uint64((1 << n) - 1)

def convert(a, thr):
    px, py, pz = (a[f"MCParticles.momentum.{c}"][:, 0] for c in "xyz")  # particle 0 = gun primary
    m = a["MCParticles.mass"][:, 0]
    assert ak.all(a["MCParticles.generatorStatus"][:, 0] == 1), "MCParticles[0] is not the gun particle"
    out = {"Ebeam": np.sqrt(px**2 + py**2 + pz**2 + m**2),   # generated energy of the gun particle
           # radius where the primary stopped existing (first inelastic interaction / conversion / exit) [mm]
           "Rend": np.hypot(a["MCParticles.endpoint.x"][:, 0], a["MCParticles.endpoint.y"][:, 0])}
    cid, e = a[f"{EC}.cellID"], a[f"{EC}.energy"]
    cryo, typ, lay = field(cid, 4, 1), field(cid, 5, 3), field(cid, 11, 8)
    # virtual detector: eDep = total energy of each escaping track; count kinetic energy for baryons, total otherwise
    # ponytail: antibaryon annihilation energy ignored
    Ev, pv = a[VD + ".eDep"], np.sqrt(sum(a[f"{VD}.momentum.{c}"]**2 for c in "xyz"))
    mv = np.sqrt(np.clip(Ev**2 - pv**2, 0, None))
    out["Evd"] = ak.sum(ak.where(mv > 0.9, Ev - mv, Ev), axis=1)
    out["Etrue"] = out["Ebeam"] - out["Evd"]                         # as in the paper: E_beam - E_VD
    act = (cryo == 0) & (typ == 0)
    E = e[act] / ak.Array(SF)[lay[act]]                             # EM-scale cell energy [GeV]
    x, y, z = (a[f"{EC}.position.{c}"][act] for c in "xyz")
    keep = E > thr
    E, x, y, z = E[keep], x[keep], y[keep], z[keep]
    r = np.sqrt(x**2 + y**2 + z**2)
    assert ak.all((np.sqrt(x**2 + y**2) > 2000) & (np.sqrt(x**2 + y**2) < 2800)), \
        "ECAL hit radii outside the barrel: sim-hit positions are not global cell positions in this release"
    # shower frame: axis n = origin -> energy-weighted cog; d = depth along n, (u,v) transverse (u ~ phi, v ~ theta)
    w = ak.sum(E, axis=1)
    cx, cy, cz = (ak.sum(E * c, axis=1) / w for c in (x, y, z))
    cn = np.sqrt(cx**2 + cy**2 + cz**2); nx, ny, nz = cx / cn, cy / cn, cz / cn
    ct = np.sqrt(nx**2 + ny**2); ux, uy = -ny / ct, nx / ct
    out.update(hit_E=E, hit_d=x * nx + y * ny + z * nz - R0 / ct, hit_u=x * ux + y * uy,
               hit_v=x * (-nz * nx / ct) + y * (-nz * ny / ct) + z * ct)
    return out, w > 0

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("out"); p.add_argument("inp", nargs="+")
    p.add_argument("--thr", type=float, default=0.003, help="cell threshold [GeV, EM scale]; tune to the real noise")
    args = p.parse_args()
    keys = uproot.open(args.inp[0])["events"].keys()
    VD = next(k for k in keys if "VD" in k and k.endswith(".eDep"))[:-5]   # collection name set by the escape counter
    br = [f"MCParticles.momentum.{c}" for c in "xyz"] + ["MCParticles.mass", "MCParticles.generatorStatus", "MCParticles.endpoint.x", "MCParticles.endpoint.y", VD + ".eDep", f"{EC}.cellID",
          f"{EC}.energy"] + [f"{EC}.position.{c}" for c in "xyz"] + [f"{VD}.momentum.{c}" for c in "xyz"]
    with uproot.recreate(args.out) as f:
        for a in uproot.iterate([i + ":events" for i in args.inp], br, step_size="200 MB"):
            out, ok = convert(a, args.thr)
            out = {k: v[ok] for k, v in out.items()}
            f["events"].extend(out) if "events" in f else f.__setitem__("events", out)
