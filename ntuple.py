#!/usr/bin/env python3
"""Raw Geant4 output of ./sim -> flat 'events' tree used by compensation.py and gnn.py.
  python ntuple.py calib calib.json e_raw_*.root              # per-layer EM-scale calibration from an electron run
  python ntuple.py make  out.root calib.json pi_raw_*.root    # calibrated hits in the shower frame  [--thr GeV]"""
import argparse, json, numpy as np, awkward as ak, uproot

# same constants as sim.cc (k4geo ALLEGRO_o1_v03)
RMIN, ALPHA, DPHI = 2172.8, np.radians(50.18), 2 * np.pi / 1536
LAYLEN = 10 * np.array([2.33596, 4.75685, 4.89843, 5.04000, 5.20989, 5.36562, 5.54966, 5.73371, 5.94607, 6.15843, 6.39909])
THGRID, THOFF, MERGE_TH = 0.009817477 / 4, 0.5902785, np.array([4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4])
SMID = np.cumsum(LAYLEN) - LAYLEN / 2                                   # layer centre along the electrode [mm]
RLAY = np.sqrt(RMIN**2 + SMID**2 + 2 * RMIN * SMID * np.cos(ALPHA))     # ... and its radius
DLAY = np.arcsin(SMID * np.sin(ALPHA) / RLAY)                           # azimuth shift of the inclined electrode there

def lut(table, idx):   # per-layer lookup on a jagged index array
    return ak.unflatten(np.asarray(table)[ak.to_numpy(ak.flatten(idx))], ak.num(idx))

def calib(inp):
    """inverse sampling fraction per layer: all deposits / LAr deposits, summed over the electron sample"""
    tot, act = np.zeros(11), np.zeros(11)
    for a in uproot.iterate([i + ":events" for i in inp], ["Elayer_all", "cell_layer", "cell_E"]):
        tot += ak.to_numpy(ak.sum(a["Elayer_all"], axis=0))
        act += np.bincount(ak.to_numpy(ak.flatten(a["cell_layer"])), ak.to_numpy(ak.flatten(a["cell_E"])), 11)
    return (tot / act).tolist()

def convert(a, c, thr):
    lay = a["cell_layer"]
    E = a["cell_E"] * lut(c, lay) * 1e-3                           # GeV, EM scale
    r, phi = lut(RLAY, lay), (2 * a["cell_module"] + 0.5) * DPHI + lut(DLAY, lay)
    theta = THOFF + (a["cell_theta"] + (lut(MERGE_TH, lay) - 1) / 2) * THGRID
    x, y, z = r * np.cos(phi), r * np.sin(phi), r / np.tan(theta)
    k = E > thr
    E, x, y, z = E[k], x[k], y[k], z[k]
    w = ak.sum(E, axis=1)
    out = {"Ebeam": a["Ebeam"] * 1e-3, "Evd": a["Evd"] * 1e-3, "Eside": a["Eside"] * 1e-3, "Edep": a["Edep"] * 1e-3,
           "Rend": a["Rend"],                                          # radius where the primary ended [mm]
           "hit_E": E, "hit_d": x - RMIN,                               # beam along x: depth, then cog-centred transverse
           "hit_u": y - ak.sum(E * y, axis=1) / w, "hit_v": z - ak.sum(E * z, axis=1) / w}
    out["Etrue"] = out["Ebeam"] - out["Evd"]                            # as in the paper: E_beam - E_VD
    return {k: v[w > 0] for k, v in out.items()}

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=["calib", "make"]); p.add_argument("args", nargs="+")
    p.add_argument("--thr", type=float, default=0.003, help="cell threshold [GeV, EM scale]; tune to the real noise")
    a = p.parse_args()
    if a.mode == "calib":
        c = calib(a.args[1:]); json.dump(c, open(a.args[0], "w")); print("1/SF per layer:", np.round(c, 3))
    else:
        c = json.load(open(a.args[1]))
        with uproot.recreate(a.args[0]) as f:
            for arr in uproot.iterate([i + ":events" for i in a.args[2:]], step_size="200 MB"):
                out = convert(arr, c, a.thr)
                f["events"].extend(out) if "events" in f else f.__setitem__("events", out)
