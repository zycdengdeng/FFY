# -*- coding: utf-8 -*-
"""两个补充分析:① HWI 直接进腿长公式  ② 觅食层位(腿的用途) vs 腿长"""
import numpy as np, pandas as pd
U="/mnt/user-data/uploads/FFY/FFY"
D=pd.read_csv(f"{U}/data/birdtree/pgls_data_full.csv"); D=D[D.HWI.notna()].copy()
FS=pd.read_csv(f"{U}/data/avonet_joined_all.csv")
FS["tipkey"]=FS.scientificNameStd.str.replace(" ","_")
D["nm"]=D.tip
J=D.merge(FS[["tipkey","ForStrat.watbelowsurf","ForStrat.wataroundsurf","ForStrat.ground",
   "ForStrat.understory","ForStrat.midhigh","ForStrat.canopy","ForStrat.aerial"]],
   left_on="nm",right_on="tipkey",how="left")
print(f"[数据] {len(D)} 种 · 有觅食层位 {J['ForStrat.ground'].notna().sum()} 种")

def ols(X,y,names):
    X=np.column_stack([np.ones(len(y))]+list(X.T if X.ndim>1 else [X]))
    b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b
    s2=r@r/(len(y)-X.shape[1]); se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
    r2=1-(r@r)/((y-y.mean())@(y-y.mean()))
    return b,se,b/se,r2

print("\n"+"="*66)
print("① 把 HWI 直接放进腿长公式（此前只做过 HWI vs 残差 u）")
print("="*66)
y=D.logL.values
b,se,t,r2=ols(D[["log_m"]].values,y,["log_m"])
print(f"  现用（只有体重）: log10 L1 = {b[0]:.4f} + {b[1]:.4f}·log10 m      R² = {r2:.3f}")
b,se,t,r2=ols(D[["log_m","HWI"]].values,y,["log_m","HWI"])
print(f"  加入 HWI 后    : log10 L1 = {b[0]:.4f} + {b[1]:.4f}·log10 m {b[2]:+.5f}·HWI   R² = {r2:.3f}")
print(f"                   系数 ±SE: b = {b[1]:.4f}±{se[1]:.4f} (t={t[1]:.0f})"
      f"   c = {b[2]:.5f}±{se[2]:.5f} (t={t[2]:.0f})")
print(f"  → 体重指数从 {0.3908:.3f} 降到 {b[1]:.3f}；HWI 每 +10，腿长 ×{10**(10*b[2]):.3f}"
      f"（即 {100*(10**(10*b[2])-1):+.1f}%）")
print(f"  → 麻雀(HWI 17)→水鸟(HWI 44)差 27：腿长 ×{10**(27*b[2]):.3f}"
      f"（{100*(10**(27*b[2])-1):+.0f}%）")

print("\n"+"="*66)
print("② 觅食层位 = 腿的用途")
print("="*66)
K={"水域":["ForStrat.watbelowsurf","ForStrat.wataroundsurf"],"地面":["ForStrat.ground"],
   "树上":["ForStrat.understory","ForStrat.midhigh","ForStrat.canopy"],"空中":["ForStrat.aerial"]}
S=J.dropna(subset=["ForStrat.ground"]).copy()
for k,cs in K.items(): S[k]=S[cs].sum(axis=1)
S["主用途"]=S[list(K)].idxmax(axis=1)
X=np.column_stack([np.ones(len(S)),S.log_m])
S["e"]=S.u.values-X@np.linalg.lstsq(X,S.u.values,rcond=None)[0]
print(f"\n  {'主要觅食层位':<8} {'n':>5} {'腿长残差':>9} {'HWI':>7} {'体重g':>8}")
for k in ["水域","地面","树上","空中"]:
    s=S[S.主用途==k]
    print(f"  {k:<8} {len(s):>5} {s.e.median():>+9.2f} {s.HWI.median():>7.1f} {10**s.log_m.median():>8.0f}")
print("\n  各层位比例与腿长残差的偏相关（控制体重）:")
for k,cs in K.items():
    v=S[k].values.astype(float)
    Z=np.column_stack([np.ones(len(S)),S.log_m])
    rv=v-Z@np.linalg.lstsq(Z,v,rcond=None)[0]
    r=np.corrcoef(rv,S.e)[0,1]; tt=r*np.sqrt((len(S)-3)/(1-r*r))
    print(f"    {k:<6} r = {r:+.3f}  (t = {tt:+.0f})")
S[["tip","主用途","e","HWI","log_m","水域","地面","树上","空中"]].to_csv(
    "/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/pgls/forstrat.csv",index=False)
