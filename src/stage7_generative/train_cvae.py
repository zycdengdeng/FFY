"""条件 VAE:工况 c → 设计族 x。生成阶段 v1 主模型。

结构(小而稳,数据几千对、x 才 3 维):
  编码器 [x(3)+c(4)] → 64 → 64 → (μ, logσ²), z 2 维
  解码器 [z(2)+c(4)] → 64 → 64 → sigmoid → x∈[0,1]³(再反归一化)
损失 = 重构 MSE + β·KL(β 线性预热,防后验坍缩)

用法: python src/stage7_generative/train_cvae.py \
        --data outputs/gen_data/gen_dataset.npz --out outputs/gen_model
产出: cvae.pt + train_log.json。CPU 即可(分钟级);A100 上加 --device cuda 更快。
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn

ZDIM = 2


class CVAE(nn.Module):
    def __init__(self, xd=3, cd=4, z=ZDIM, h=64):
        super().__init__()
        self.zdim = z
        self.enc = nn.Sequential(nn.Linear(xd + cd, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU())
        self.mu = nn.Linear(h, z); self.lv = nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z + cd, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, xd), nn.Sigmoid())

    def forward(self, x, c):
        e = self.enc(torch.cat([x, c], -1))
        mu, lv = self.mu(e), self.lv(e).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(torch.cat([z, c], -1)), mu, lv

    @torch.no_grad()
    def sample(self, c, n):
        """c:(cd,) 归一化;返回 (n, xd) 归一化设计。"""
        cc = c.unsqueeze(0).expand(n, -1)
        z = torch.randn(n, self.zdim)
        return self.dec(torch.cat([z, cc], -1))


def norm(v, lo, hi):
    return (v - lo) / (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/gen_data/gen_dataset.npz")
    ap.add_argument("--out", default="outputs/gen_model")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--zdim", type=int, default=2, help="7维设计建议 3-4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    d = np.load(args.data)
    meta = json.load(open(args.data.replace("gen_dataset.npz", "gen_dataset_meta.json")))
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    x_lo, x_hi = np.array(meta["x_lo"]), np.array(meta["x_hi"])
    C = torch.tensor(norm(d["C_tr"], c_lo, c_hi), dtype=torch.float32)
    X = torch.tensor(norm(d["X_tr"], x_lo, x_hi), dtype=torch.float32)
    print(f"[cvae] train pairs {len(C)}  device {args.device}")

    model = CVAE(xd=X.shape[1], cd=C.shape[1], z=args.zdim).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ds = torch.utils.data.TensorDataset(X, C)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    log = []
    for ep in range(args.epochs):
        beta = args.beta * min(1.0, ep / max(args.epochs * 0.3, 1))   # KL 预热
        mse_s = kl_s = nb = 0
        for xb, cb in dl:
            xb, cb = xb.to(args.device), cb.to(args.device)
            xh, mu, lv = model(xb, cb)
            mse = ((xh - xb) ** 2).mean()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).mean()
            loss = mse + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            mse_s += mse.item(); kl_s += kl.item(); nb += 1
        log.append(dict(ep=ep, mse=mse_s / nb, kl=kl_s / nb, beta=beta))
        if (ep + 1) % 50 == 0:
            print(f"  ep {ep+1}/{args.epochs}  mse={mse_s/nb:.5f} kl={kl_s/nb:.3f}")

    torch.save(dict(state=model.state_dict(), meta=meta,
                    xd=X.shape[1], zdim=args.zdim),
               os.path.join(args.out, "cvae.pt"))
    json.dump(log, open(os.path.join(args.out, "train_log.json"), "w"))
    print(f"[cvae] saved → {args.out}/cvae.pt")


if __name__ == "__main__":
    main()
