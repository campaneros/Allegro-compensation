import argparse, os, numpy as np, awkward as ak, uproot
from scipy.optimize import curve_fit

EB = np.arange(5, 101, 5)          # beam-energy bins [GeV], as in the paper
rng = np.random.default_rng(1)

def load(fn):
    a = uproot.open(fn)["events"].arrays()
    E, w = a["hit_E"], ak.sum(a["hit_E"], axis=1)
    d = {"Ebeam": a["Ebeam"], "Etrue": a["Etrue"], "Ereco": w,
         "rms": np.sqrt(ak.sum(E * (a["hit_u"]**2 + a["hit_v"]**2), axis=1) / w),   # eq. 4.1 (hits are cog-centred)
         "zcog": ak.sum(E * a["hit_d"], axis=1) / w}                                  # eq. 4.3
    d = {k: ak.to_numpy(v).astype(float) for k, v in d.items()}
    d["Ehc"] = np.clip(d["Ebeam"] - d["Etrue"], 0, None)   # ideal "HCAL" = everything not deposited in the ECAL
    return d

def ebin(d): return np.digitize(d["Ebeam"], EB) - 1

def fit_corr(d, var, nq=12):
    """per beam-energy bin: straight line through the profile of Ereco/Etrue vs var (eq. 4.2)"""
    # ponytail: profile = median in quantile bins, the paper fits the peak of each slice; switch if tails bias it
    resp, b, out = d["Ereco"] / d["Etrue"], ebin(d), {}
    for i in range(len(EB) - 1):
        m = (b == i) & np.isfinite(resp) & np.isfinite(d[var])
        if m.sum() < 20 * nq: continue
        q = np.quantile(d[var][m], np.linspace(0, 1, nq + 1)); k = np.clip(np.digitize(d[var][m], q) - 1, 0, nq - 1)
        out[i] = np.polyfit([np.median(d[var][m][k == j]) for j in range(nq)],
                            [np.median(resp[m][k == j]) for j in range(nq)], 1)
    return out

def apply_corr(d, var, par):
    b, E = ebin(d), d["Ereco"].copy()
    for i, (p1, p0) in par.items():
        m = b == i
        E[m] /= np.clip(p0 + p1 * d[var][m], 0.2, None)   # clip: never scale up by more than x5
    return E

def smear(d, s):  # Gaussian HCAL with stochastic term s on the energy reaching it
    return d["Ehc"] + s * np.sqrt(d["Ehc"]) * rng.standard_normal(len(d["Ehc"]))

def reso(y, d):
    """sigma68 (half width of the 16-84% interval) of y/Ebeam per beam-energy bin"""
    b, x, r, e = ebin(d), [], [], []
    for i in range(len(EB) - 1):
        v = (y / d["Ebeam"])[b == i]
        if len(v) < 100: continue
        lo, hi = np.percentile(v, [16, 84])
        x.append((EB[i] + EB[i + 1]) / 2); r.append((hi - lo) / 2); e.append((hi - lo) / 2 / np.sqrt(2 * len(v)) * 1.3)
    return np.array(x), np.array(r), np.array(e)

def nsc(E, N, S, C): return np.sqrt((N / E)**2 + S**2 / E + C**2)
def fit_nsc(x, r, e):
    if len(x) < 4: return np.full(3, np.nan), np.full(3, np.nan)   # not enough energy bins for a 3-parameter fit
    p, cov = curve_fit(nsc, x, r, sigma=e, p0=[1, .5, .02], bounds=(0, np.inf), absolute_sigma=True)
    return p, np.sqrt(np.diag(cov))

def toy(fn, n=40000):
    """fake pion showers: wide hadronic part with response 0.7, narrow EM part with response 1"""
    Eb = rng.uniform(5, 100, n); cont = rng.beta(2, 2, n); Et = Eb * cont; fem = rng.beta(2, 2.5, n)
    nh = 80; em = rng.random((n, nh)) < 0.4
    w = rng.exponential(1, (n, nh)); w = np.where(em, w / (w * em).sum(1, keepdims=True) * fem[:, None],
                                                 w / (w * ~em).sum(1, keepdims=True) * (1 - fem[:, None]) * 0.7)
    sig = np.where(em, 15, 70)
    hit = {"hit_E": w * Et[:, None], "hit_u": rng.normal(0, sig), "hit_v": rng.normal(0, sig),
           "hit_d": np.where(em, rng.uniform(50, 200, (n, nh)), rng.uniform(100, 450, (n, nh)))}
    with uproot.recreate(fn) as f: f["events"] = {"Ebeam": Eb, "Etrue": Et, **{k: ak.from_regular(v) for k, v in hit.items()}}

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("train", nargs="?"); p.add_argument("test", nargs="?")
    p.add_argument("--gnn", help="npy with GNN-predicted ECAL energy [GeV] for test.root (from gnn.py)")
    p.add_argument("--hcal", type=float, default=0.25, help="HCAL stochastic term"); p.add_argument("-o", default="plots")
    p.add_argument("--ebin", type=float, default=5, help="width of the beam-energy bins [GeV]; widen it for small samples")
    p.add_argument("--toy"); a = p.parse_args()
    EB = np.arange(5, 100 + a.ebin / 2, a.ebin)
    if a.toy: toy(a.toy); raise SystemExit
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(a.o, exist_ok=True)
    tr = load(a.train); te = load(a.test) if a.test else tr
    nbin = np.bincount(ebin(te)[(ebin(te) >= 0) & (ebin(te) < len(EB) - 1)], minlength=len(EB) - 1)
    print(f"train {len(tr['Ebeam'])} events, test {len(te['Ebeam'])} events, {nbin.min()}-{nbin.max()} test events per {a.ebin:g} GeV bin")
    if (nbin >= 100).sum() == 0:
        raise SystemExit("No energy bin has the 100 test events needed for a resolution: simulate more events or widen the bins with --ebin")
    if (nbin >= 100).sum() < 4: print("fewer than 4 usable energy bins: resolutions are plotted, the N/S/C fit is skipped")
    ecal = {"no correction": te["Ereco"]}
    for var, lab in (("rms", "RMS correction"), ("zcog", "Z correction")):
        ecal[lab] = apply_corr(te, var, fit_corr(tr, var))
    if a.gnn: ecal["GNN"] = np.load(a.gnn)
    hc = smear(te, a.hcal)

    # Figs. 4 / 7: response vs shower-shape variable
    for var, xl in (("rms", "ECAL cluster RMS [mm]"), ("zcog", "ECAL weighted depth [mm]")):
        plt.figure(); m = np.isfinite(te[var]); resp = te["Ereco"] / te["Etrue"]
        plt.hist2d(te[var][m], resp[m], bins=(80, 80), range=((0, np.percentile(te[var][m], 99.5)), (0, 1.6)), cmin=1)
        p1, p0 = np.polyfit(te[var][m & (resp < 2)], resp[m & (resp < 2)], 1)
        xx = np.array(plt.xlim()); plt.plot(xx, p0 + p1 * xx, "r", label=f"{p0:.3f} {p1:+.4f} x"); plt.legend()
        plt.xlabel(xl); plt.ylabel("E_reco / E_true (ECAL)"); plt.savefig(f"{a.o}/response_vs_{var}.png"); plt.close()

    # Figs. 5 / 9: (E_ECAL + E_HCAL)/E_beam - 1 in two energy slices
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for axi, (lo, hi) in zip(ax, ((20, 25), (75, 80))):
        m = (te["Ebeam"] > lo) & (te["Ebeam"] <= hi)
        for lab, E in ecal.items(): axi.hist(((E + hc) / te["Ebeam"] - 1)[m], bins=80, range=(-.6, .6), histtype="step", label=lab)
        axi.set_title(f"{lo} < E_beam < {hi} GeV"); axi.set_xlabel("(E_ECAL + E_HCAL)/E_beam - 1"); axi.legend()
    fig.savefig(f"{a.o}/distributions.png"); plt.close(fig)

    # Figs. 6 / 8 / 10 / 11 + Tables 1-2: combined resolution vs energy, N/S/C fit
    plt.figure(); xx = np.linspace(EB[0], EB[-1], 200)
    print(f"combined ECAL + {a.hcal:.0%}/sqrt(E) HCAL          N [GeV]        S [%]         C [%]")
    table = {}
    for lab, E in ecal.items():
        x, r, e = reso(E + hc, te); par, err = fit_nsc(x, r, e); table[lab] = r
        print(f"{lab:38s} {par[0]:.2f}+-{err[0]:.2f}   {100*par[1]:.1f}+-{100*err[1]:.1f}   {100*par[2]:.2f}+-{100*err[2]:.2f}")
        l = plt.errorbar(x, r, e, fmt="o", ms=3, label=f"{lab}: S = {100*par[1]:.1f}%"); plt.plot(xx, nsc(xx, *par), "--", c=l[0].get_color())
    print("\nsigma68/E_beam [%] per energy bin (no fit involved):\n" + f"{'E_beam [GeV]':>16s}" + "".join(f"{v:8.1f}" for v in x))
    for lab, r in table.items(): print(f"{lab:>16s}" + "".join(f"{100*v:8.2f}" for v in r))
    print()
    plt.xlabel("E_beam [GeV]"); plt.ylabel("sigma68 / E_beam"); plt.ylim(0); plt.legend(); plt.savefig(f"{a.o}/resolution.png"); plt.close()

    # Fig. 12: ECAL contribution = combined (-) HCAL-only, for several HCAL stochastic terms
    plt.figure(); best = list(ecal)[-1]
    for s in (.25, .30, .35, .40, .45, .50):
        h = smear(te, s); x, rc, e = reso(ecal[best] + h, te); _, rh, _ = reso(te["Etrue"] + h, te)
        c = np.sqrt(np.clip(rc**2 - rh**2, 0, None)); plt.errorbar(x, c, e, fmt="o-", ms=3, label=f"HCAL {s:.0%}/sqrt(E)")
    par, err = fit_nsc(x, c, e)
    print(f"ECAL contribution ({best}): N={par[0]:.2f} GeV  S={100*par[1]:.1f}%  C={100*par[2]:.2f}%")
    plt.xlabel("E_beam [GeV]"); plt.ylabel(f"ECAL contribution / E_beam ({best})"); plt.ylim(0); plt.legend()
    plt.savefig(f"{a.o}/ecal_contribution.png"); plt.close()
