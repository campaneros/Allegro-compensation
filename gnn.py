#!/usr/bin/env python3
"""ParticleNet-style regression of the true ECAL energy from hits (x, y, z, E), as in arXiv:2606.05111 sec. 5.
  python gnn.py train.root test.root pred.npy [--epochs 60]"""
import argparse, numpy as np, awkward as ak, uproot, torch, torch.nn as nn

SCALE = 30.  # target = Etrue / SCALE, same convention as pred_energy in the CRILIN analysis code

def load(fn, nmax):
    a = uproot.open(fn)["events"].arrays()
    o = ak.argsort(a["hit_E"], ascending=False)
    f = [ak.to_numpy(ak.fill_none(ak.pad_none(a[k][o], nmax, clip=True), 0)).astype(np.float32) * s
         for k, s in (("hit_u", .01), ("hit_v", .01), ("hit_d", .01), ("hit_E", 1.))]
    x = torch.tensor(np.stack(f, -1))
    return x, x[..., 3] > 0, torch.tensor(ak.to_numpy(a["Etrue"]).astype(np.float32) / SCALE)

class EdgeConv(nn.Module):
    def __init__(s, cin, cout, k):
        super().__init__(); s.k = k; s.sc = nn.Linear(cin, cout)
        # ponytail: no BatchNorm (masking it is fiddly); add it if training is unstable
        s.mlp = nn.Sequential(nn.Linear(2 * cin, cout), nn.ReLU(), nn.Linear(cout, cout), nn.ReLU(), nn.Linear(cout, cout), nn.ReLU())
    def forward(s, pts, x, mask):
        d = torch.cdist(pts, pts).masked_fill(~mask[:, None, :], float("inf"))
        idx = d.topk(min(s.k + 1, x.shape[1]), largest=False).indices[..., 1:]      # k nearest, minus self
        B = torch.arange(x.shape[0], device=x.device)[:, None, None]
        nb, nm = x[B, idx], mask[B, idx][..., None]
        xi = x[:, :, None].expand_as(nb)
        e = (s.mlp(torch.cat([xi, nb - xi], -1)) * nm).sum(2) / nm.sum(2).clamp(min=1)
        return torch.relu(e + s.sc(x))

class Net(nn.Module):
    def __init__(s, k=50, drop=0.3):
        super().__init__()
        s.convs = nn.ModuleList([EdgeConv(4, 32, k), EdgeConv(32, 32, k), EdgeConv(32, 64, k)])
        s.head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))
    def forward(s, x, mask):
        pts = x[..., :3]                       # first block: neighbours in space; then in feature space
        for c in s.convs: x = c(pts, x, mask); pts = x
        m = mask[..., None]
        return s.head((x * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("train"); p.add_argument("test"); p.add_argument("out")
    p.add_argument("--epochs", type=int, default=60); p.add_argument("--k", type=int, default=50)
    p.add_argument("--nmax", type=int, default=256, help="keep the nmax most energetic hits"); a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x, m, y = load(a.train, a.nmax); nv = len(y) // 5                      # 20% validation
    net = Net(a.k).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-4); best = float("inf")
    def run(x, m, y=None, train=False, bs=64):
        net.train(train); out, perm = [], torch.randperm(len(x)) if train else torch.arange(len(x))
        for i in range(0, len(x), bs):
            j = perm[i:i + bs]
            with torch.set_grad_enabled(train):
                pr = net(x[j].to(dev), m[j].to(dev))
                if train: opt.zero_grad(); nn.functional.mse_loss(pr, y[j].to(dev)).backward(); opt.step()
            out.append(pr.detach().cpu())
        return torch.cat(out)
    for ep in range(a.epochs):
        run(x[nv:], m[nv:], y[nv:], train=True)
        val = nn.functional.mse_loss(run(x[:nv], m[:nv]), y[:nv]).item(); print(f"epoch {ep} val mse {val:.5f}", flush=True)
        if val < best: best = val; torch.save(net.state_dict(), a.out + ".pt")
    net.load_state_dict(torch.load(a.out + ".pt"))
    x, m, _ = load(a.test, a.nmax); np.save(a.out, run(x, m).numpy() * SCALE)
