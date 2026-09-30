# -*- coding: utf-8 -*-
"""P2/P3 预备:从每个骨架的 1000 棵树里定种子抽 100 棵,存成多行 newick。
R 侧 ape::read.tree 可直接读。种子固定 42,可复现。"""
import io, sys, zipfile, random
ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
BT = f"{ROOT}/data/birdtree"
N, SEED = 100, 42
for tag, zp in (("hackett", "HackettStage2_0001_1000.zip"),
                ("ericson", "EricsonStage2_0001_1000.zip")):
    keep = set(random.Random(SEED).sample(range(1000), N))
    out, kept = f"{BT}/{tag}_100.nwk", 0
    with zipfile.ZipFile(f"{BT}/{zp}") as z, open(out, "w") as w:
        with z.open(z.namelist()[0]) as f:
            for i, line in enumerate(io.TextIOWrapper(f, "utf-8", errors="replace")):
                if i in keep and line.strip():
                    w.write(line if line.endswith("\n") else line + "\n"); kept += 1
    print(f"[{tag}] 抽 {kept}/{N} 棵 → {out}")
