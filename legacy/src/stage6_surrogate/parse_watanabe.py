"""解析 Watanabe 2017 (The Auk 134:672) 附录表 10:103 种现生雁鸭科骨骼实测。

来源:Watanabe, J. 2017. Quantitative discrimination of flightlessness in
fossil Anatidae from skeletal proportions. The Auk 134:672-695.
附录表 10:物种均值(mm),FEM=股骨 TIB=胫跗骨 TMT=跗跖骨。

输出 data/skeletal/watanabe2017_anatidae.csv:
species, group, fem_mm, tib_mm, tmt_mm, n_min  (仅保留三骨齐全的行)
并计算 r2=TIB/TMT, r3=FEM/TMT。

用法: python src/stage6_surrogate/parse_watanabe.py --pdf <path> [--out data/skeletal]
"""
import argparse
import csv
import os
import re

import pdfplumber

ROW = re.compile(
    r"^(?P<name>[A-Z][A-Za-z.\- ]+?)\s*(?P<grp>Volant|Flightless)\s+(?P<vals>.+)$")
VAL = re.compile(r"(?:(\d+(?:\.\d+)?)\s*\((\d+)\)|–)")


def parse(pdf_path):
    doc = pdfplumber.open(pdf_path)
    rows, cur_genus = [], None
    for pno in (21, 22):
        for line in (doc.pages[pno].extract_text() or "").split("\n"):
            m = ROW.match(line.strip())
            if not m:
                continue
            name = re.sub(r"\s+", " ", m.group("name")).strip()
            # 修 PDF 粘连:CygnusatratusVolant 类;补全缩写属名
            name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
            for glued, fixed in [("Cygnusatratus", "Cygnus atratus"),
                                 ("Cairinamoschata", "Cairina moschata")]:
                name = name.replace(glued, fixed)
            if re.match(r"^[A-Z]\.", name):          # "C. olor" / "B.c.interior"
                if cur_genus:
                    name = cur_genus + " " + name.split(".", 1)[1].strip(". ")
            else:
                cur_genus = name.split()[0]
            vals = VAL.findall(m.group("vals"))
            if len(vals) != 7:                        # CAR HUM ULN CMC FEM TIB TMT
                continue
            fem, tib, tmt = vals[4], vals[5], vals[6]
            if not (fem[0] and tib[0] and tmt[0]):
                continue                              # 三骨不齐,弃
            rows.append(dict(
                species=name, group=m.group("grp"),
                fem_mm=float(fem[0]), tib_mm=float(tib[0]), tmt_mm=float(tmt[0]),
                n_min=min(int(fem[1]), int(tib[1]), int(tmt[1]))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="/mnt/user-data/uploads/FFY/文献调研_视频到3D重建/"
                    "Quantitative discrimination of flightlessness in fossil Anatidae.pdf")
    ap.add_argument("--out", default="data/skeletal")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = parse(args.pdf)
    for r in rows:
        r["r2"] = round(r["tib_mm"] / r["tmt_mm"], 3)
        r["r3"] = round(r["fem_mm"] / r["tmt_mm"], 3)
    fp = os.path.join(args.out, "watanabe2017_anatidae.csv")
    with open(fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    import numpy as np
    vol = [r for r in rows if r["group"] == "Volant"]
    r2 = np.array([r["r2"] for r in vol]); r3 = np.array([r["r3"] for r in vol])
    tmt = np.array([r["tmt_mm"] for r in vol])
    print(f"parsed {len(rows)} species ({len(vol)} volant, 三骨齐全)")
    print(f"TMT(L1): {tmt.min():.0f}-{tmt.max():.0f} mm  p5-p95 {np.percentile(tmt,5):.0f}-{np.percentile(tmt,95):.0f}")
    print(f"r2=TIB/TMT: {r2.min():.2f}-{r2.max():.2f}  p5-p95 {np.percentile(r2,5):.2f}-{np.percentile(r2,95):.2f}  中位 {np.median(r2):.2f}")
    print(f"r3=FEM/TMT: {r3.min():.2f}-{r3.max():.2f}  p5-p95 {np.percentile(r3,5):.2f}-{np.percentile(r3,95):.2f}  中位 {np.median(r3):.2f}")
    for sp in ["Cygnus olor", "Anser anser", "Branta canadensis canadensis", "Anas platyrhynchos"]:
        hit = [r for r in rows if r["species"].startswith(sp.split()[0]) and sp.split()[-1] in r["species"]]
        for h in hit[:1]:
            print(f"  {h['species']}: TMT={h['tmt_mm']} r2={h['r2']} r3={h['r3']}")
    print("saved:", fp)


if __name__ == "__main__":
    main()
