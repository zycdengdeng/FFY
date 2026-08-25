"""真实推理:bio 终版模型在 5 个体重下生成→实摔→选优,存下全部几何与结构数据。"""
import sys, json
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, '/tmp/pipeline_code/src/stage10_v2')
sys.path.insert(0, '/tmp/pipeline_code/src/stage7_generative')
import physics_v2 as P
from bioprior import BioPrior
from factory_v2 import zeta_of_kc
from e17_emergent_b import load
from train_cvae import norm

KC, V0, GCAP, SMAX = 1.0e5, 1.2, 10*9.81, 0.024

def ev(a):
    x, m = a
    r = P.eval_v2(tuple(x), m, V0, kc=KC, zeta_c=zeta_of_kc(KC), npass=2)
    if r is None or r.get("fail"):
        return None
    ok, _ = P.feasible_v2(r, GCAP, SMAX)
    return dict(ok=bool(ok), peak_g=r["peak_a"]/9.81, stroke_mm=r["leg_stroke_mm"],
                D_mm=r["D_mm"], leg_mass_g=r["leg_mass_kg"]*1e3,
                mass_frac=r["mass_frac"]) if ok or True else None

if __name__ == "__main__":
    U = "/mnt/user-data/uploads/FFY/FFY/outputs/"
    model, meta = load(U + "v2_e5_bio/cvae_r40.pt")
    prior = BioPrior("bio", sigma=meta["prior"]["sigma"], u_max=meta["prior"]["u_max"])
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    out = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for m in (1.0, 2.0, 4.0, 8.0, 12.0):
            torch.manual_seed(7)
            c = np.array([np.log10(m), V0, np.log10(KC), GCAP, SMAX])
            cn = torch.tensor(norm(c, c_lo, c_hi), dtype=torch.float32)
            X = prior.expand(np.clip(model.sample(cn, 20).numpy(), 0, 1), m)
            rs = list(ex.map(ev, [(x, m) for x in X], chunksize=2))
            cand = [(r, x) for r, x in zip(rs, X) if r and r["ok"]]
            if not cand:
                cand = [(r, x) for r, x in zip(rs, X) if r]
            r, x = min(cand, key=lambda t: t[0]["peak_g"])
            L1 = float(x[0]); l = [L1, float(x[1])*L1, float(x[2])*L1]
            out.append(dict(m=m, L_mm=l, D_mm=r["D_mm"], x7=[float(v) for v in x],
                            peak_g=r["peak_g"], stroke_mm=r["stroke_mm"],
                            leg_mass_g=r["leg_mass_g"], mass_frac=r["mass_frac"],
                            n_feas=sum(1 for q in rs if q and q["ok"])))
            print(f"m={m:>4}: L={l[0]:.0f}/{l[1]:.0f}/{l[2]:.0f}mm  D={['%.1f'%d for d in [0] and r['D_mm']]}  "
                  f"peak {r['peak_g']:.2f}g  腿重 {r['leg_mass_g']:.0f}g  可行 {sum(1 for q in rs if q and q['ok'])}/20", flush=True)
    json.dump(out, open("/tmp/designs.json", "w"), indent=2)
    print("saved /tmp/designs.json")
