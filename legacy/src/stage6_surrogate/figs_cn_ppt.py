# -*- coding: utf-8 -*-
"""中文版 PPT 配图：直接复用英文脚本的绘图代码，只替换显示字符串 + 换中文字体。
这样中英两版布局完全一致，数字也一定同源（都从 CSV 现算）。
产出目录 outputs/ppt_cn/，文件名前缀 cn_。"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
U = os.path.dirname(os.path.dirname(HERE))

FONT = '''
import matplotlib.font_manager as _fm
_have={f.name for f in _fm.fontManager.ttflist}
_cjk=next((f for f in ("Noto Sans CJK SC","Noto Sans CJK JP","WenQuanYi Zen Hei","Droid Sans Fallback")
           if f in _have), None)
plt.rcParams["font.sans-serif"]=([_cjk] if _cjk else [])+["DejaVu Sans"]
plt.rcParams["font.family"]="sans-serif"
plt.rcParams["axes.unicode_minus"]=False
'''

# 只替换「显示用」字符串，不碰标识符 —— 每条都足够独特
MAP = [
 # ---- F1 异速生长 ----
 ("waterbirds, ±1 SD band  (SD = {sd:.3f} dex)", "水鸟 ±1σ 带（σ = {sd:.3f} dex）"),
 ("waterbirds   $b$ = {b_p:.3f}   (n = {nsp})", "水鸟   $b$ = {b_p:.3f}   （n = {nsp} 种）"),
 ("all birds   $b$ = {b[1]:.3f}", "全部鸟类   $b$ = {b[1]:.3f}"),
 ("log$_{10}$  body mass  [g]", "log$_{10}$ 体重  [g]"),
 ("log$_{10}$  tarsometatarsus $L_1$  [mm]", "log$_{10}$ 跗跖长 $L_1$  [mm]"),
 ("Law 1 — leg length scales with body mass\\n{n:,} bird species", "定律一 · 腿长随体重标度\\n{n:,} 个鸟类物种"),
 ("standardised residual   $u$", "标准化残差   $u$"),
 ('B.set_ylabel("species"', 'B.set_ylabel("物种数"'),
 ("The residual is the signal\\n", "残差才是信号\\n"),
 ("u < 0 : shorter leg than the prior\\n           expects for that mass\\n",
  "u < 0：比先验对该体重的预期更短\\n"),
 ("u > 0 : longer leg", "u > 0：比预期更长"),
 # ---- F2 HWI ----
 ("body mass only", "只用体重"),
 ("body mass + flight efficiency (HWI)", "体重 + 飞行效率 HWI"),
 ('A.set_xlabel("predicted  log$_{10}L_1$", fontsize=12.5)', 'A.set_xlabel("公式预测的  log$_{10}L_1$", fontsize=12.5)'),
 ('ax[0].set_ylabel("observed  log$_{10}L_1$"', 'ax[0].set_ylabel("实测  log$_{10}L_1$"'),
 ("HWI +10  →  leg  {d10:+.0f}%\\n", "HWI 每 +10  →  腿长 {d10:+.0f}%\\n"),
 ("sparrow (17) → waterbird (44)  →  leg  {d_sw:+.0f}%", "麻雀(17) → 水鸟(44)  →  腿长 {d_sw:+.0f}%"),
 ("Law 2 — flight efficiency belongs in the leg-length law   (n = {n:,})",
  "定律二 · 飞行效率应当进入腿长公式   （n = {n:,}）"),
 # ---- F3 PGLS ----
 ("OLS\\n(ignores phylogeny)", "普通最小二乘\\n（不管亲缘）"),
 ("PGLS\\n(200 trees)", "PGLS\\n（200 棵树）"),
 (r'r"$\beta_{\rm HWI}$   (effect on leg residual $u$)"', r'"$\\beta_{\\rm HWI}$   （对腿长残差 $u$ 的效应）"'),
 ("Shared ancestry explains\\n{100*(1-med/b_ols):.0f}% of the naive effect",
  "共同祖先解释掉了\\n朴素效应的 {100*(1-med/b_ols):.0f}%"),
 ('B.set_ylabel("trees", fontsize=12)', 'B.set_ylabel("树的棵数", fontsize=12)'),
 ('C.set_ylabel("trees", fontsize=12)', 'C.set_ylabel("树的棵数", fontsize=12)'),
 ("…but it never vanishes\\n200 trees, all p < {R.p.max():.0e}", "但它从不消失\\n200 棵树，全部 p < {R.p.max():.0e}"),
 ("(0 = no phylogenetic signal, 1 = Brownian)", "（0 = 无亲缘信号，1 = 布朗运动）"),
 ("Is it just shared ancestry?  —  phylogenetic generalised least squares, ",
  "这只是共同祖先造成的吗？ —— 系统发育广义最小二乘，"),
 ("f\"{len(R)} trees × 2 backbones, n = {int(R.n.iloc[0]):,} species\"",
  "f\"{len(R)} 棵树 × 2 个骨架，n = {int(R.n.iloc[0]):,} 个物种\""),
 # ---- F46 独立重复 ----
 ('A.set_xlabel(f"{nt} families (≥15 species each), sorted", fontsize=11.5)',
  'A.set_xlabel(f"{nt} 个科（每科 ≥15 种），按差值排序", fontsize=11.5)'),
 ("high-HWI half − low-HWI half\\nleg residual  [SD]", "腿长残差之差  [标准差]\\n高飞行效率半 − 低飞行效率半"),
 ("(a)  Inside one family:  the better-flying half has shorter legs",
  "(a)  同一个科内部：飞得更好的那一半，腿更短"),
 ("{ns}/{nt} = {100*ns/nt:.0f}% of families      ", "{ns}/{nt} = {100*ns/nt:.0f}% 的科      "),
 ("f\"(50% expected by chance;  z = {z:.1f})\"", "f\"（若无关系应为 50%；z = {z:.1f}）\""),
 ("stricter, within genus:  {gs}/{gt} = {100*gs/gt:.0f}%   (z = {gz:.1f})",
  "更严：属内比较  {gs}/{gt} = {100*gs/gt:.0f}%   （z = {gz:.1f}）"),
 ("shift in leg residual\\nat the clade's stem  $\\\\Delta u$  [SD]", "支系干上的腿长残差跳变\\n$\\\\Delta u$  [标准差]"),
 ("(b)  Across the tree:  {tot} mutually non-nested clades that changed leg length independently",
  "(b)  跨整棵树：{tot} 个互不嵌套的支系，各自独立改变了腿长"),
 ("clade mean HWI vs $\\\\Delta u$ :   r = {rr:+.3f}", "支系平均 HWI 与 $\\\\Delta u$ 的相关：r = {rr:+.3f}"),
 ("shortened {sh}   ·   lengthened {tot-sh}", "变短 {sh} 个   ·   变长 {tot-sh} 个"),
 ("The trade-off is replicated independently at two scales — it is not one ancient accident",
  "这条权衡在两个尺度上各自独立地重演 —— 不是一次古老的偶然"),
 # ---- F5 觅食层位 ----
 ('MAP = {"水域": "Water", "地面": "Ground", "树上": "Trees", "空中": "Aerial"}',
  'MAP = {"水域": "水域", "地面": "地面", "树上": "树上", "空中": "空中"}'),
 ('order = ["Aerial", "Water", "Trees", "Ground"]', 'order = ["空中", "水域", "树上", "地面"]'),
 ('"\\u7a7a\\u4e2d", "\\u6c34\\u57df", "\\u6811\\u4e0a", "\\u5730\\u9762"',
  '"\\u7a7a\\u4e2d", "\\u6c34\\u57df", "\\u6811\\u4e0a", "\\u5730\\u9762"'),
 ("leg-length residual  (mass removed)  [SD]", "腿长残差（已扣体重）[标准差]"),
 ("What is the leg actually used for?", "这条腿到底是干什么用的？"),
 ("aerial feeders → shortest legs\\nground feeders → longest legs", "空中取食 → 腿最短\\n地面取食 → 腿最长"),
 ("partial correlation:  % of foraging done in that stratum\\n",
  "偏相关：在该层位觅食的百分比\\n"),
 ("vs leg-length residual   (body mass held constant)", "与腿长残差（已固定体重）"),
 ("Leg length tracks what the leg does\\nall {len(S):,} species", "腿长跟着腿的用途走\\n全部 {len(S):,} 个物种"),
]
EXTRA = [  # 管线图 / A0 图
 ("The dataset we built  —  nothing new was measured; the value is in the join",
  "我们搭的数据集 —— 没有新测任何数据，价值在这次拼接"),
 ("\"Tobias et al. 2022, Ecol Lett\\n\"\n"
  "   \"11,009 species x 11 morphological traits,\\n\"\n"
  "   \"measured on museum skins:\\n\"\n"
  "   \"tarsus, wing, Kipp's distance, mass\"",
  "\"Tobias et al. 2022, Ecol Lett\\n\"\n"
  "   \"11,009 个物种 × 11 项形态测量，\\n\"\n"
  "   \"取自博物馆标本：\\n\"\n"
  "   \"跗跖、翼长、Kipp 距离、体重\""),
 ("\"Wilman et al. 2014, Ecology\\n\"\n"
  "   \"diet (10 categories) and foraging\\n\"\n"
  "   \"stratum (7 strata) for every\\n\"\n"
  "   \"extant bird species\"",
  "\"Wilman et al. 2014, Ecology\\n\"\n"
  "   \"每个现生鸟种的食性（10 类）\\n\"\n"
  "   \"与觅食层位（7 层）\\n\"\n"
  "   \"均已编码\""),
 ("\"Jetz et al. 2012, Nature\\n\"\n"
  "   \"9,993 tips, 2 backbones; we draw\\n\"\n"
  "   \"100 trees per backbone to carry\\n\"\n"
  "   \"topological uncertainty\"",
  "\"Jetz et al. 2012, Nature\\n\"\n"
  "   \"9,993 个末端、2 个骨架；每个骨架\\n\"\n"
  "   \"抽 100 棵树，把拓扑\\n\"\n"
  "   \"不确定性一起带上\""),
 ('family=("DejaVu Sans Mono" if mono else "DejaVu Sans")',
  'family=plt.rcParams["font.sans-serif"][0]'),
 ("scaling  $b_{eff}=0.238$", "标度  $b_{eff}=0.238$"),
 ("columns we actually use", "真正用到的列"),

 ('bird families (≥15 species each), sorted', '个科（每科 ≥15 种），按差值排序'),
 ('leg-length residual:', '腿长残差之差：'),
 ('high-HWI half − low-HWI half  [SD]', '高飞行效率半 − 低飞行效率半  [标准差]'),
 ('Independent replication — inside a single family, the better fliers have shorter legs', '独立重复 —— 在同一个科内部，飞得更好的那一半，腿更短'),
 ('shorter legs:  {ns} families', '腿更短：{ns} 个科'),
 ('longer legs:  {nt-ns} families', '腿更长：{nt-ns} 个科'),
 ('{ns}/{nt} = {100*ns/nt:.0f}% of families', '{ns}/{nt} = {100*ns/nt:.0f}% 的科'),
 ('(50% expected by chance;  z = {z:.1f},  p < 1e-8)', '（若无关系应为 50%；z = {z:.1f}，p < 1e-8）'),
 ('Method: within each family, remove the body-mass effect,', '做法：每个科内部先扣掉体重的影响，'),
 ('split species into high / low HWI halves, compare residuals.', '再把该科物种按飞行效率分成高、低两半，比较腿长残差。'),
 ('Same family ⇒ near-identical ancestry and lifestyle,', '同科物种亲缘极近、生活方式相似 ——'),
 ('so the difference can only come from flight efficiency.', '差别只能来自飞行效率本身。'),

 ('species name on the tree', '树上的物种名'),
 ('tarsometatarsus length (mm)', '跗跖长 (mm)'),
 ('body mass (g)', '体重 (g)'),
 ("hand-wing index = Kipp's distance / wing length", '手翼指数 = Kipp 距离 / 翼长'),
 ('the standard proxy for flight efficiency', '飞行效率的标准代理量'),
 ('standardised leg-length residual', '标准化的腿长残差'),
 ('% of foraging done in water / ground / trees / air', '在水域/地面/树上/空中觅食的百分比'),
 ('clade membership, for the within-clade tests', '所属科与目，用于族内检验'),
 ('The biology does not design the leg.  It tells the generator where to look —', '生物并不设计这条腿。它只告诉生成器该往哪里找 ——'),
 ("and the physics then tells us where biology's answer stops being the right one for a 30 kg machine.", '然后由物理告诉我们：对一台 30 kg 的机器，生物给的答案从哪里开始不再是对的。'),
 ('tarsometatarsus length      <- the biological prior sets this', '跗跖长        ← 生物先验定的就是这一维'),
 ('segment ratios ', '分段比 '),
 ('touchdown ankle / knee angle', '触地时的踝角 / 膝角'),
 ('   dimensionless joint stiffness', '   无量纲关节刚度'),
 ('relaxation time  $c/k$', '松弛时间  $c/k$'),
 ('airframe mass          5 - 30 kg', '机身质量      5 – 30 kg'),
 ('sink rate               0.6 - 2.4 m/s', '下沉速度      0.6 – 2.4 m/s'),
 ('terrain stiffness      soil - concrete', '地面刚度      软土 – 混凝土'),
 ('acceptance limits      10 g,  24 mm', '验收上限      10 g,  24 mm'),
 ('Ablation over 3 architectures, 15/15 conditions covered by each:', '三个架构做消融，每个都覆盖全部 15/15 工况：'),
 ('peak deceleration  3.60 / 3.52 / 3.49 g  -  differences below the', '峰值过载 3.60 / 3.52 / 3.49 g —— 差异低于换随机种子'),
 ('seed-to-seed noise floor (2.9%).  So performance does not pick the', '自身造成的噪声地板（2.9%）。既然性能分不出架构，'),
 ('architecture; manufacturability does.  A0 drops one shock unit for free', '那就由制造性来分：A0 白拿掉一套减震器，还拿到 36 mm'),
 ('and gains a 36 mm femur lever arm -> an off-the-shelf spring.', '股骨力臂 → 可以直接买货架弹簧。'),
 ('peak deceleration    ', '峰值过载    '),
 ('        stroke   ', '        落震行程   '),
 ('slenderness  and  leg-mass budget   satisfied', '细长比与腿重预算  均满足'),
 ('evaluated by an Exudyn multibody drop simulation', '由 Exudyn 多体落震仿真判定'),

 ("our merged table", "拼出来的总表"),
 ("name crosswalk\\nBirdLife <-> BirdTree\\n", "命名对照\\nBirdLife ↔ BirdTree\\n"),
 ("% matched\\n\\n", "% 对上\\n\\n"),
 ("{n:,} species\\n{nfam} families · {nord} orders\\n\\n", "{n:,} 个物种\\n{nfam} 科 · {nord} 目\\n\\n"),
 ("flightless taxa removed", "已剔除不会飞的类群"),
 ('("$L_1$","tarsometatarsus length (mm)",BLU)', '("$L_1$","跗跖长 (mm)",BLU)'),
 ('("$m$","body mass (g)",BLU)', '("$m$","体重 (g)",BLU)'),
 ('"hand-wing index = Kipp\'s distance / wing length\\n"\n             "the standard proxy for flight efficiency"',
  '"手翼指数 = Kipp 距离 / 翼长\\n"\n             "飞行效率的标准代理量"'),
 ('"standardised leg-length residual\\n"', '"标准化的腿长残差\\n"'),
 ('("ForStrat","% of foraging done in water / ground / trees / air",GRN)',
  '("ForStrat","在水域/地面/树上/空中觅食的百分比",GRN)'),
 ('("Family, Order","clade membership, for the within-clade tests",GRY)',
  '("Family, Order","所属科与目，用于族内检验",GRY)'),
 ('"8,705 birds"', '"8,705 只鸟"'),
 ('"the leg-length law\\n"', '"腿长定律\\n"'),
 ('"phylogenetically\\ncorrected, 200 trees"', '"做过亲缘校正\\n200 棵树"'),
 ('"design prior"', '"设计先验"'),
 ('"for a target mass:\\n"\n        "a 1-sigma corridor of\\nleg lengths, not one\\nnumber\\n\\n"\n        "the search space is\\nnarrowed, not fixed"',
  '"给定目标质量：\\n"\n        "得到一条 1σ 的腿长走廊，\\n而不是一个数\\n\\n"\n        "搜索空间被收窄，\\n不是被钉死"'),
 ('"generator"', '"生成器"'),
 ('"conditional model\\n"', '"条件生成模型\\n"'),
 ('"trained on Exudyn\\ndrop simulations\\nof A0 legs"', '"在 A0 腿的 Exudyn\\n落震仿真上训练"'),
 ('"feasibility"', '"可行性判据"'),
 ('"10 g  /  24 mm\\nslenderness\\nleg-mass budget\\n\\n"\n        "every candidate is\\nsimulated, not scored\\nby a surrogate"',
  '"10 g  /  24 mm\\n细长比\\n腿重预算\\n\\n"\n        "每个候选都真跑仿真，\\n不是只由代理模型\\n打分"'),
 ('"what emerges"', '"涌现出来的"'),
 ('"the generator\'s own\\n"', '"生成器自己的\\n"'),
 ('"flatter than biology:\\nthe machine affords\\nstiffer joints than\\na bird can"',
  '"比生物更平：机器能\\n用鸟做不到的关节\\n刚度和阻尼"'),
 ("The biology does not design the leg.  It tells the generator where to look —\\n"
  "and the physics then tells us where biology's answer stops being the right one for a 30 kg machine.",
  "生物并不设计这条腿。它只告诉生成器该往哪里找 ——\\n"
  "然后由物理告诉我们：对一台 30 kg 的机器，生物给的答案从哪里开始不再是对的。"),
 ("From a biological law to a machine  —  where the prior actually enters",
  "从一条生物定律到一台机器 —— 先验究竟从哪里进来"),
 # A0
 ("A0 — the mechanical abstraction we simulate", "A0 —— 我们真正拿去仿真的力学抽象"),
 ('"ankle"', '"踝"'), ('"knee"', '"膝"'), ('"hip"', '"髋"'),
 ("$L_1$  tarsometatarsus", "$L_1$  跗跖骨"),
 ("$L_2 = r_2 L_1$  tibiotarsus", "$L_2 = r_2 L_1$  胫跗骨"),
 ("$L_3 = r_3 L_1$  femur", "$L_3 = r_3 L_1$  股骨"),
 ('"ankle shock"', '"踝减震器"'),
 ('"hip shock"', '"髋减震器"'),
 ("knee: free hinge\\n(no spring, no damper)", "膝：自由铰\\n无弹簧、无阻尼"),
 ('"airframe  $m$"', '"机身  $m$"'),
 ("ground: Hertz–Kelvin contact,  $k_c$", "地面：Hertz–Kelvin 接触，$k_c$"),
 ("Design vector  x  (9 dimensions)", "设计向量 x（9 维）"),
 ("$L_1$            tarsometatarsus length      <- the biological prior sets this\\n"
  "$r_2$, $r_3$          segment ratios $L_2/L_1$, $L_3/L_1$\\n"
  "$\\\\theta_A$, $\\\\theta_K$        touchdown ankle / knee angle\\n"
  "$\\\\kappa_{ankle}$, $\\\\kappa_{knee}$, $\\\\kappa_{hip}$   dimensionless joint stiffness\\n"
  "$\\\\tau$             relaxation time  $c/k$",
  "$L_1$            跗跖长          ← 生物先验定的就是这一维\\n"
  "$r_2$, $r_3$          分段比 $L_2/L_1$, $L_3/L_1$\\n"
  "$\\\\theta_A$, $\\\\theta_K$        触地时的踝角 / 膝角\\n"
  "κ踝, κ膝, κ髋       无量纲关节刚度\\n"
  "$\\\\tau$             松弛时间  $c/k$"),
 ("Operating condition  z  (5 dimensions)", "工况向量 z（5 维）"),
 ("$\\\\log_{10} m$   airframe mass          5 - 30 kg\\n"
  "$v_0$        sink rate               0.6 - 2.4 m/s\\n"
  "$\\\\log_{10} k_c$  terrain stiffness      soil - concrete\\n"
  "$g_{cap}$, $s_{max}$  acceptance limits      10 g,  24 mm",
  "$\\\\log_{10} m$   机身质量        5 – 30 kg\\n"
  "$v_0$        下沉速度        0.6 – 2.4 m/s\\n"
  "$\\\\log_{10} k_c$  地面刚度        软土 – 混凝土\\n"
  "$g_{cap}$, $s_{max}$  验收上限        10 g,  24 mm"),
 ("Why A0 and not the textbook 3-shock leg", "为什么选 A0，而不是教科书式的三减震器腿"),
 ("Ablation over 3 architectures, 15/15 conditions covered by each:\\n"
  "peak deceleration  3.60 / 3.52 / 3.49 g  -  differences below the\\n"
  "seed-to-seed noise floor (2.9%).  So performance does not pick the\\n"
  "architecture; manufacturability does.  A0 drops one shock unit for free\\n"
  "and gains a 36 mm femur lever arm -> an off-the-shelf spring.",
  "三个架构做消融，每个都覆盖全部 15/15 工况：\\n"
  "峰值过载 3.60 / 3.52 / 3.49 g —— 差异低于换随机种子\\n"
  "自身造成的噪声地板（2.9%）。既然性能分不出架构，\\n"
  "那就由制造性来分：A0 白拿掉一套减震器，还拿到 36 mm\\n"
  "股骨力臂 → 可以直接买货架弹簧。"),
 ("Feasibility (what the generator must satisfy)", "可行性判据（生成器必须满足的）"),
]

def run(src_name, out_dir_new, prefix_new, extra=()):
    cand = [os.path.join(HERE, src_name),
            os.path.join(os.path.dirname(HERE), "stage10_v2", src_name)]
    path = next(c for c in cand if os.path.exists(c))
    src = open(path, encoding="utf-8").read()
    for a, b in list(MAP) + list(extra):
        src = src.replace(a, b)
    src = src.replace('OUT = f"{U}/outputs/ppt_en"', f'OUT = f"{{U}}/outputs/{out_dir_new}"')
    src = src.replace('OUT=f"{U}/outputs/ppt_en"',   f'OUT=f"{{U}}/outputs/{out_dir_new}"')
    src = src.replace('out="outputs/ppt_en/en_F8_a0.png"', f'out="outputs/{out_dir_new}/{prefix_new}F8_a0.png"')
    src = src.replace('os.makedirs("outputs/ppt_en"', f'os.makedirs("outputs/{out_dir_new}"')
    src = re.sub(r'/en_(F\d+[A-Za-z0-9_]*\.png)', f'/{prefix_new}\\1', src)
    # 字体：插在 rcParams 设置之后
    src = src.replace('plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,\n'
                      '                     "savefig.facecolor": "white", "figure.facecolor": "white"})',
                      'plt.rcParams.update({"savefig.facecolor": "white", "figure.facecolor": "white"})' + FONT)
    src = src.replace('plt.rcParams.update({"font.family":"DejaVu Sans","axes.unicode_minus":False,\n'
                      '                     "savefig.facecolor":"white","figure.facecolor":"white"})',
                      'plt.rcParams.update({"savefig.facecolor":"white","figure.facecolor":"white"})' + FONT)
    src = src.replace('plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})',
                      'plt.rcParams.update({})' + FONT)
    g = {"__name__": "__main__", "__file__": path}
    exec(compile(src, src_name + "[cn]", "exec"), g)

os.makedirs(f"{U}/outputs/ppt_cn", exist_ok=True)
run("figs_en_ppt.py",    "ppt_cn", "cn_", EXTRA)   # EXTRA 里对不上的条目是 no-op,安全
run("fig_pipeline_en.py","ppt_cn", "cn_", EXTRA)
# fig_a0_en.py 已改成自带 --lang，不再走字符串替换
print("→", f"{U}/outputs/ppt_cn")
