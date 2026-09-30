# -*- coding: utf-8 -*-
"""本周英文汇报 PPT（13 页）—— 直接建在 EIT 模板上（沿用其母版/版式/配色/字体）。
模板来源: 每周汇报/09_09_2026_Week5_International_Meeting_EN.pptx
模板约定:
  标题幻灯片版式  -> 首页（标题 32pt 白 微软雅黑 @(0.666,2.712)，副标题 20pt 白 @(3.59,5.23)）
  空白版式        -> 内页（携带 EIT logo/装饰）
  标题文本框      -> (0.401,0.254) 11.60x0.505  24pt 粗 #8E2A34 微软雅黑
  副标题条        -> (0.401,0.972) 12.18x0.4375 20pt 粗 #7F2D32（本稿用 16pt，避免换行）
  底部圆角条      -> (0.652,6.605) 12.03x0.717  实心 #7F2D32，白字 20pt（本稿 15pt）
"""
import os, copy
from pptx import Presentation
from pptx.util import Inches as I, Pt, Emu
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

U = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F, AN = f"{U}/outputs/ppt_en", f"{U}/outputs/anim_b"
TPL = f"{U}/每周汇报/09_09_2026_Week5_International_Meeting_EN.pptx"
OUT = f"{U}/每周汇报/07_09_2026_Week7_Group_Meeting_EN.pptx"
CRIM, CRIM2, WHITE = C(0x8E, 0x2A, 0x34), C(0x7F, 0x2D, 0x32), C(0xFF, 0xFF, 0xFF)
INK, GOLD = C(0x1A, 0x1A, 0x1A), C(0xFF, 0xC0, 0x00)
YH = "微软雅黑"

prs = Presentation(TPL)
# --- 清空模板里的旧页，只留母版/版式 ---
xml = prs.slides._sldIdLst
for sid in list(xml):
    prs.part.drop_rel(sid.rId); xml.remove(sid)
LAY = {l.name: l for l in prs.slide_masters[0].slide_layouts}
TITLE_LAY, BLANK_LAY = LAY["标题幻灯片"], LAY["空白"]

def txt(sl, x, y, w, h, s, size, bold, color, align=PP_ALIGN.LEFT, space=1.0, anchor=None):
    t = sl.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0); tf.margin_top = tf.margin_bottom = Emu(0)
    if anchor: tf.vertical_anchor = anchor
    for i, ln in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln; p.alignment = align; p.line_spacing = space
        for r in p.runs:
            r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
            r.font.name = YH
            r.font._rPr.set("altLang", "en-US")
    return t

def page(title, kicker):
    """内页：模板标准标题 + 副标题条"""
    sl = prs.slides.add_slide(BLANK_LAY)
    txt(sl, 0.401, 0.254, 9.85, 0.505, title, 24, True, CRIM)   # 9.85 避开右上角 EIT logo
    if kicker: txt(sl, 0.401, 0.900, 12.18, 0.60, kicker, 15, True, CRIM2, space=1.12)
    return sl

def bottom(sl, s, size=14):
    """底部圆角结论条（模板配色）"""
    sh = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(0.652), I(6.605), I(12.03), I(0.717))
    sh.fill.solid(); sh.fill.fore_color.rgb = CRIM2
    sh.line.fill.background(); sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = I(0.12)
    for i, ln in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln; p.alignment = PP_ALIGN.CENTER; p.line_spacing = 1.05
        for r in p.runs:
            r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = YH
    return sh

BOX = (0.40, 1.62, 12.53, 4.86)          # 内页图片可用区
def fig(sl, path, box=BOX):
    x, y, w, h = box
    iw, ih = Image.open(path).size; ar = iw/ih
    ww, hh = (h*ar, h) if w/h > ar else (w, w/ar)
    return sl.shapes.add_picture(path, I(x+(w-ww)/2), I(y+(h-hh)/2), I(ww), I(hh))

def notes(sl, s): sl.notes_slide.notes_text_frame.text = s

# ================= 1 首页（模板标题版式） =================
sl = prs.slides.add_slide(TITLE_LAY)
for ph in list(sl.placeholders):
    ph._element.getparent().remove(ph._element)
txt(sl, 0.666, 2.35, 12.38, 1.50,
    "Can 8,705 Birds Tell Us How Long a Landing Gear Should Be?", 32, True, WHITE, space=1.15)
txt(sl, 0.666, 3.95, 12.38, 0.55,
    "A phylogenetically controlled leg-length law — and where the machine disagrees with it",
    17, False, WHITE)
txt(sl, 3.59, 5.23, 9.45, 1.10,
    "Week 7 · SJTU-EIT PhD · Zihan Wang · Advisors: Yifan Zhao, K. Jimmy Hsia\n7 September 2026",
    18, False, WHITE, space=1.30)
notes(sl, "15 分钟。这次讲的是「我们在做什么」，不是「我们做了多少」。两半："
          "(1) 我们从鸟身上抽出来的一条定律，以及我们怎么想尽办法去推翻它；"
          "(2) 把这条定律交给机器之后，机器在哪一个数上不同意生物。")

# ================= 2 数据集 =================
sl = page("The Dataset We Built",
          "Three published sources, one join — and a phylogeny attached to every row.")
fig(sl, f"{F}/en_F7_dataset.png")
bottom(sl, "8,705 species · 188 families · 35 orders, each carrying morphology, ecology "
           "and a position on the avian tree.")
notes(sl, "AVONET 给形态，EltonTraits 给生态，BirdTree 给亲缘。BirdLife 与 BirdTree 两套命名法不同，"
          "需要 crosswalk，98.6% 对上。不会飞的类群剔除。HWI = Kipp 距离 / 翼长，是这个文献里"
          "飞行效率的标准代理量：翼尖越尖、展弦比越高，长距离飞行越省力。")

# ================= 3 Law 1 =================
sl = page("Step 1 — Fit the Law, Keep the Residual",
          "Classical allometry. What we keep is the part it does NOT explain: the residual u.")
fig(sl, f"{F}/en_F1_allometry.png")
bottom(sl, "Design prior = the phylogenetically corrected waterbird fit:  b = 0.365,  σ = 0.081 dex.  "
           "Everything after this slide is about u.")
notes(sl, "左：8,705 种。橙色虚线是不管亲缘直接回归的结果；绿线是我们真正取先验的类群——水鸟，"
          "在生态上最接近一台落在软而不平地面上的重型 VTOL——并且把亲缘协方差放进误差项里拟合。"
          "b=0.365 表示腿长比体重长得慢；几何相似应当是 1/3，所以略高于等比。"
          "右：减掉这条线、再除以散度，得到 u，即「同体重下腿长的 z 分数」。u<0 = 腿比该体重"
          "应有的更短。后面所有生物结论的因变量都是 u。")

# ================= 4 设计先验带 =================
sl = page("What the Prior Actually Looks Like",
          "Law 1 back on the raw scale, over the mass range we design for: a centre line and a ±2.5σ corridor.")
fig(sl, f"{F}/en_F_prior.png", (0.40, 1.62, 12.53, 4.86))
bottom(sl, "This corridor is what the generator receives: for a target mass, a range of plausible leg "
           "lengths rather than a single number.  Above 12 kg there is no waterbird — that part is extrapolated.")
notes(sl, "This is the prior we hand the generator, drawn on the raw scale. The red line is the waterbird "
          "fit, b = 0.366, with a ±2.5σ corridor around it. Waterbirds are the clade we look at most closely: "
          "they land on soft, uneven ground at low speed, and they are also where our perception pipeline "
          "gets its footage. The two grey lines are theory — geometric similarity at 1/3 and elastic "
          "similarity at 1/4 — and birds are steeper than both, so leg length grows faster than pure "
          "geometric scaling would predict. Green marks the 5–30 kg product range. The honest part: above "
          "12 kg there is no waterbird — swans and pelicans stop there — so the top half of our range is "
          "extrapolation, and that is exactly why the heavy end has to be settled by simulation rather than "
          "by birds.")

# ================= 5 Law 2 =================
sl = page("Step 2 — Flight Efficiency Enters the Law",
          "HWI = hand-wing index = Kipp's distance / wing length.  High HWI = efficient, long-range flight.")
fig(sl, f"{F}/en_F2_hwi.png")
bottom(sl, "R² 0.671 → 0.793 from a single term.  A sparrow-to-waterbird change in wing shape "
           "predicts a 31% shorter leg at the same body mass.")
notes(sl, "这是我们之前没做过的一版拟合：把 HWI 直接放进腿长公式，而不是只对残差 u 做回归。"
          "多这一项，R² 从 0.67 提到 0.79——吸收掉纯体重定律留下的 37% 方差。系数 −0.0061/HWI，"
          "t=−71。可以直接读成一个权衡：为高效飞行投资的鸟，要拿腿来付账。"
          "这正是起落架设计者面对的同一笔交易——行程 vs 带着走的质量。")

# ================= 5 PGLS =================
sl = page("Step 3 — Is It Just Shared Ancestry?",
          "Closely related species are not independent data points.  PGLS, 200 trees × 2 backbones.")
fig(sl, f"{F}/en_F3_pgls.png", (0.40, 1.68, 12.53, 4.70))
bottom(sl, "Phylogeny eats 65% of the naive effect — and the remaining 35% survives every "
           "single tree (p < 1e-56, λ = 0.93).")
notes(sl, "这页是给生物学家看的。普通最小二乘假设残差独立，但在系统树上并不独立——两只雨燕像，"
          "是因为共同祖先，不是因为各自独立解出了同一个答案。PGLS 把树带来的期望协方差放进误差项。"
          "λ=0.93 说明这个性状几乎完全被亲缘结构化，所以修正很狠：系数从 −0.078 掉到 −0.027。"
          "但 200 棵树里没有一棵让它碰到 0。我们报的是缩水后站得住的那个值，不是好看的那个。")

# ================= 7 隼科具体例子 =================
sl = page("What the Trade-off Looks Like — Two Falcons",
          "Before any statistics: the easiest case to read. Same family, near-identical ancestry, both raptors, both fly.")
fig(sl, f"{F}/en_F_falcon.png", (0.40, 1.58, 12.53, 4.92))
bottom(sl, "The forest-falcon is 65 g lighter than the sooty falcon and still carries 81% more leg.  "
           "Across all 62 Falconidae, flight efficiency and leg residual correlate at r = −0.68.")
notes(sl, "Slow down here — this is the easiest slide in the talk. The sooty falcon is a long-distance "
          "migrant that hunts on the wing in open sky; the leg is pure dead weight and drag, and its "
          "tarsometatarsus is 34 mm. The forest-falcon threads the Amazon canopy in short bursts and has to "
          "perch and grasp; its tarsometatarsus is 62 mm — while being 65 g lighter. Both are Falconidae: "
          "same ancestry, same diet, both fly, so the difference cannot be blamed on ancestry or on a wildly "
          "different lifestyle. On the right, all 62 species of the family, r = −0.68. Split the family at "
          "HWI = 47 and the halves differ by −1.48 — that is the Falconidae bar in the next figure.")

# ================= 8 科内重复 =================
sl = page("Step 4a — The Tree Argues With Itself: inside families",
          "If this were one ancient accident, it would not have to happen again inside every family.")
fig(sl, f"{F}/en_F4_within.png", (0.40, 1.58, 12.53, 4.92))
bottom(sl, "77 of 96 families repeat it internally (50% expected by chance, z = 5.9); "
           "tightened to genus, still 251 of 419 (60%, z = 4.1).")
notes(sl, "This is the falcon story repeated 96 times. Inside each family, remove the mass effect, split into "
          "better- and worse-flying halves, compare. Same ancestry, same lifestyle — the only thing left is "
          "flight efficiency. 80% of families go the predicted way. The green box is the stricter version: no "
          "crossing even a genus boundary, and 251 of 419 genera still go the same way.")

# ================= 9 跨支系跳变 =================
sl = page("Step 4b — The Tree Argues With Itself: across the tree",
          "A different scale: not inside families, but which clades changed leg length on their own stem.")
fig(sl, f"{F}/en_F_shifts.png", (0.40, 1.58, 12.53, 4.92))
bottom(sl, "24 mutually non-nested, independent shifts: 10 of the 12 better-flying clades shortened, only 6 "
           "of the 12 poorer-flying ones; shift size vs clade flight efficiency, r = −0.59.")
notes(sl, "Each bar is a shift in leg length on one clade's own stem; the grey line is that clade's mean "
          "flight efficiency. Only 24 mutually non-nested clades are kept — I first counted 40, but 16 sit "
          "inside another, so the same evolutionary event was counted twice. Removing the double-counting "
          "made the correlation stronger, from −0.40 to −0.59. The left half is dominated by blue shortening, "
          "the right half by red lengthening. The exceptions are worth naming: grebes and Pelecaniformes both "
          "went positive — diving and wading clades, where the leg has another job in the water. That is what "
          "the next slide is about.")

# ================= 10 功能检验 =================
sl = page("Step 5 — A Function Check",
          "If the law is real, leg length should track how the leg is used — not only how the wing is shaped.")
fig(sl, f"{F}/en_F5_forstrat.png", (0.40, 1.68, 12.53, 4.70))
bottom(sl, "Aerial feeders carry the shortest legs (−2.6 SD), ground feeders the longest (+0.9 SD); "
           "the partial correlation flips sign exactly where function says it should.")
notes(sl, "EltonTraits 给出每个物种在各个觅食层位的百分比。在空中取食的鸟几乎不用腿，腿就极短；"
          "在地面行走觅食的鸟需要腿，也就要为它付钱。控制体重后，地面觅食比例与腿长残差的偏相关"
          "+0.29，树上 −0.21。水域最弱，只有 −0.06，因为水鸟本身很不均质——涉水的鹭和潜水的鸬鹚"
          "都算「水域」。这是这一页诚实的保留项。")

# ================= 11 管线 =================
sl = page("Handing the Law to the Machine",
          "The prior does not choose the design. It chooses the region the generator is allowed to search.")
fig(sl, f"{F}/en_F9_pipeline.png", (0.40, 1.72, 12.53, 4.62))
bottom(sl, "A 1σ corridor, not a number — so the physics can still tell the generator 'no'.")
notes(sl, "对力学背景的听众，这一页的定位很重要：我们不是在抄一只鸟。定律给的是「目标质量下"
          "合理腿长的走廊」；生成器在 5 维工况条件下提出 9 维设计；每一个候选都由完整多体落震仿真"
          "校核，而不是由代理模型打分。生物负责收窄搜索，物理负责拍板。")

# ================= 12 A0 =================
sl = page("One Page of Engineering — the A0 Leg",
          "Three segments, two shock units, free knee — chosen by manufacturability, not by performance.")
fig(sl, f"{F}/en_F8_a0.png", (0.40, 1.58, 12.53, 4.92))
bottom(sl, "The architecture ablation showed all three candidates equivalent within the seed-to-seed "
           "noise floor — so A0 wins on part count and on being able to buy the spring.")
notes(sl, "三段的命名沿用鸟：跗跖骨、胫跗骨、股骨。两套减震器而不是三套：我们消融掉膝簧，峰值"
          "从 3.60 g 变到 3.52 g，低于换随机种子就能造成的 2.9%。既然性能分不出架构，那就由"
          "制造性来分：A0 在股骨上保住 36 mm 力臂，把 1385 kN/m 的髋簧变成 616 kN/m——"
          "这是「特殊订制」和「货架件」的区别。")

# ================= 13 机器自己走出这条律（核心页）=================
sl = page("Where the Machine Agrees With the Bird",
          "First, plainly: the power law is in our parameterisation, not a discovery. The independent evidence is on the right.")
fig(sl, f"{F}/en_F_agree.png", (0.40, 1.58, 12.53, 4.92))
bottom(sl, "On the segment ratios the machine matches the bird to within 1% of the search box (3 terrains × 2 seeds).  "
           "The exponent holds near the biological value on soft ground and drops to 0.263 on rigid ground — never touching the corridor wall.")
notes(sl, "占位——讲稿见文件末尾的 SCRIPT。")

# ================= 14 一次落震（视频，仅当已按新口径重跑时插入）=================
import os as _os
_MP4 = f"{AN}/b_compare_hard_en.mp4"; _PNG = f"{AN}/b_compare_hard_en_last.png"
if _os.path.exists(_MP4) and _os.path.exists(_PNG):
    sl = page("What That Difference Costs, in One Landing",
              "Same 30 kg body, same speed, rigid ground, same joints. Only the exponent differs: 0.366 against 0.263.")
    _iw, _ih = Image.open(_PNG).size
    _w = 12.0; _h = _w*_ih/_iw
    if _h > 4.86: _h = 4.86; _w = _h*_iw/_ih
    sl.shapes.add_movie(_MP4, I(0.40+(12.53-_w)/2), I(1.62+(4.86-_h)/2), I(_w), I(_h),
                        poster_frame_image=_PNG, mime_type="video/mp4")
    bottom(sl, "Next:  a rebound criterion in the acceptance test  ·  horizontal touchdown speed as an input  ·  hardware.")
    notes(sl, "占位——讲稿见文件末尾的 SCRIPT。")
else:
    print("⚠ 未找到按新口径重跑的落震视频，已跳过第 14 页；跑完 anim 再重建即可")

# ---- 备注 = 中文稿 + 分隔线 + 英文稿（改稿只改 speaker_script_cn/en.py）----
from speaker_script_cn import SCRIPT_CN  # noqa: E402
from speaker_script_en import SCRIPT     # noqa: E402
_n = len(prs.slides._sldIdLst)
assert len(SCRIPT_CN) == len(SCRIPT), "中英讲稿段数对不上"
assert _n in (len(SCRIPT), len(SCRIPT)-1), f"讲稿 {len(SCRIPT)} 段,幻灯片 {_n} 页"
_RULE = "\n\n" + "—" * 34 + "\n"
for _sl, _cn, _en in zip(prs.slides, SCRIPT_CN[:_n], SCRIPT[:_n]):
    _sl.notes_slide.notes_text_frame.text = (
        "【中文稿】\n\n" + _cn.strip() + _RULE + "【English script】\n\n" + _en.strip())

try:                                   # 文件可能正被 PowerPoint 打开
    prs.save(OUT)
except PermissionError:
    OUT = OUT.replace(".pptx", "_new.pptx")
    prs.save(OUT)
    print("⚠ 原文件被占用(多半开着 PowerPoint),已另存为 _new.pptx")
print("→", OUT, os.path.getsize(OUT)//1024, "KB, 10 slides")
