# -*- coding: utf-8 -*-
"""v3 汇报 PPT（10 页，生成优先）。中英两版同一脚本。
    python src/stage10_v2/mk_ppt_v3.py --lang cn
    python src/stage10_v2/mk_ppt_v3.py --lang en
架构页保留用户手工做的那一页：
  cn ← 当前中文版第 2 页（系统架构流程图）
  en ← Week5 国际会议版第 4 页（System Architecture Flowchart）
其余页由脚本生成；备注 = 中文稿 + 英文稿。"""
import os, sys, argparse
from pptx import Presentation
from pptx.util import Inches as I, Pt, Emu
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speaker_script_cn import SCRIPT_CN
from speaker_script_en import SCRIPT

ap = argparse.ArgumentParser(); ap.add_argument("--lang", default="cn", choices=["cn", "en"])
LANG = ap.parse_args().lang
U = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F, AN = f"{U}/outputs/ppt_{LANG}", f"{U}/outputs/anim_b"
SRC = {"cn": (f"{U}/每周汇报/07_09_2026_Week7_Group_Meeting_CN.pptx", "系统架构流程图"),
       "en": (f"{U}/每周汇报/09_09_2026_Week5_International_Meeting_EN.pptx", "System Architecture Flowchart")}[LANG]
OUT = f"{U}/每周汇报/07_09_2026_Week7_v3_{LANG.upper()}.pptx"
CRIM, CRIM2, WHITE, INK = C(0x8E,0x2A,0x34), C(0x7F,0x2D,0x32), C(0xFF,0xFF,0xFF), C(0x1A,0x1A,0x1A)
YH = "微软雅黑"
P = lambda k: f"{F}/{LANG}_{k}.png"

T = {"cn": dict(
  bg=("课题背景", "现有小型 VTOL 无人机起落架",
      [("现有中小型多旋翼与垂直起降无人机，起落架普遍采用", 0),
       ("滑橇式支架配橡胶脚垫", 1),
       ("的形式。该构型结构简单、成本低、维护方便，在农业植保、电力巡检与物流投送等场景中已大量装机。然而，与大中型飞机成熟的", 0),
       ("油气式缓冲支柱", 1),
       ("不同，这一量级的起落架几乎不含独立的缓冲机构 —— 着陆能量只能由支架自身的弹性变形与脚垫的压缩吸收。随着任务载荷不断增加、着陆场地从硬化跑道扩展到", 0),
       ("草地、湿沙与不平整地面", 1),
       ("，传统起落架逐渐暴露出三个主要问题：", 0),
       ("缓冲行程受限、着陆过载偏高、缺乏面向工况的设计方法", 1),
       ("。", 0)],
      ["缓冲行程受限", "着陆过载偏高", "缺乏设计方法"],
      "水鸟腿的多段柔顺结构，为小型 VTOL 起落架的高效缓冲设计提供了新思路"),
  t1=("8,705 只鸟能告诉我们起落架该多长吗？", "面向工况的起落架生成设计，以鸟类腿长定律为先验",
      "第七周 · 上海交大–宁波东方理工联培 · 王子晗 · 导师：赵一帆、夏焜\n2026 年 9 月 7 日"),
  arch=("问题与方法", "目标：一台按工况直接给出起落架设计的机器；每个输出都经落震仿真验证。"),
  s3=("腿长的生物先验", "201 种水鸟，log₁₀L₁ = 0.616 + 0.366·log₁₀m；±2.5σ 的走廊就是交给生成器的先验。",
      "L₁ 的先验是一段区间，不是一个值。σ = 0.081 dex；12 kg 以上无水鸟数据。"),
  s4=("飞行效率与腿长的权衡", "把手翼指数 HWI 加进腿长公式，R² 0.671 → 0.783；同体重下飞得好的腿更短。",
      "林隼较烟隼轻 65 g，L₁ 长 81%。隼科 62 种：飞行效率与腿长残差 r = −0.68。"),
  s5=("权衡在亲缘校正下依然成立", "亲缘接近的物种不是独立样本 —— 三种检验。",
      "PGLS 200 树 p < 1e-56（效应缩水 65%）· 科内 77/96 · 独立支系 10/12 对 6/12。"),
  s6=("生成器：输入、输出与验证", "5 个工况量进，9 个设计量出；每个输出在接受前都在仿真器里再摔一次。",
      "9 维设计 ← 5 维工况。生物先验仅约束 L₁ 一维，其余由力学界定。"),
  s7=("全包线内的生成设计", "三个工况 → 三条腿（同一比例尺）→ 各自的落震结果。每条都是可行集的中位，不是挑出来的。",
      "首发可行率 83–87%（19 质量 × 6 工况 × 2 种子）。真实水鸟工况下：生成 95%，实测鸟腿 71%。"),
  s8=("生成器未被告知却学到的东西", "幂律是我们写进参数化的；真正独立的证据是分段比与姿态。",
      "分段比与鸟中位差 <1% 盒宽；有效指数 软地 0.34 / 硬地 0.26，不触边界；踝角低 17°。"),
  s9=("同一只真天鹅，同一套关节：生成器改了什么", "号手天鹅 11.07 kg，腿的几何全部来自实测；生成器在同一体重下自己给一条腿。",
      "关节刚度两侧完全相同，只有几何和触地姿态不同：L₁ 108 → 122 mm、踝角 144° → 128°。峰值 7.49 → 4.19 g（低 44%），腿重 127 → 93 g（轻 27%）。"),
  s95=("与 DJI Agras T30 式起落架的对比",
       "同材料、同工况、总重相当（709 g vs 669 g）：滑橇的行程 4.7 mm、峰值 33 g；A0 行程 22.6 mm、峰值 7.9 g。"),
  s10=("总结", None,
       "鸟的腿长定律很紧，内部有飞行效率与腿长的交易，且经得起亲缘校正。\n"
       "把它以走廊而非一个数交给生成器：全包线内返回经验证的设计，首发命中率 > 80%。\n"
       "生成器偏离生物的地方 —— 硬地、姿态 —— 都有可以说出的物理原因。",
       "未完成：验收中的回弹判据 · 水平触地速度作为输入 · 硬件")),
 "en": dict(
  bg=("Background", "Landing gear on small VTOL drones today",
      [("Small and mid-size multirotor and VTOL drones almost all use a ", 0),
       ("skid frame with rubber foot pads", 1),
       (" - cheap, simple, and already flying in large numbers in agriculture, inspection and delivery. Unlike the mature ", 0),
       ("oleo-pneumatic shock strut", 1),
       (" of large aircraft, gear at this scale carries no separate shock absorber: the landing energy can only go into the frame bending and the pads squashing. As payloads grow and landing sites move from paved pads to ", 0),
       ("turf, wet sand and uneven ground", 1),
       (", three problems surface: ", 0),
       ("limited stroke, high landing deceleration, and no design method for a given operating point", 1),
       (".", 0)],
      ["limited stroke", "high deceleration", "no design method"],
      "The multi-segment compliant leg of a waterbird offers a new route to efficient shock absorption for small VTOL gear"),
  t1=("Can 8,705 Birds Tell Us How Long a Landing Gear Should Be?",
      "Operating-point-conditioned generative design of landing gear, with an avian leg-length law as prior",
      "Week 7 · SJTU–EIT PhD · Zihan Wang · Advisors: Yifan Zhao, K. Jimmy Hsia\n7 September 2026"),
  arch=("Problem and Approach", "Goal: a generator that returns a landing-gear design for a given operating point; every output verified by drop simulation."),
  s3=("A Biological Prior for Leg Length", "201 waterbird species, log₁₀L₁ = 0.616 + 0.366·log₁₀m; the ±2.5σ corridor is the prior handed to the generator.",
      "The prior on L₁ is an interval, not a value. σ = 0.081 dex; no waterbird data above 12 kg."),
  s4=("Flight Efficiency Trades Against Leg Length", "Adding the hand-wing index (HWI) to the law: R² 0.671 → 0.783; at equal mass the better flier has the shorter leg.",
      "Forest-falcon: 65 g lighter, L₁ 81% longer. Falconidae, n = 62: r = −0.68 between flight efficiency and leg residual."),
  s5=("The Trade-off Survives Phylogenetic Control", "Closely related species are not independent samples — three checks.",
      "PGLS, 200 trees, p < 1e-56 (effect shrinks 65%) · within families 77/96 · independent clades 10/12 vs 6/12."),
  s6=("The Generator: Inputs, Outputs, Verification", "Five operating-point numbers in, nine design numbers out; every output is dropped again before acceptance.",
      "9-dim design ← 5-dim operating point. The biological prior constrains L₁ only; the rest is bounded by mechanics."),
  s7=("Generated Designs Across the Operating Envelope", "Three operating points → three legs (common scale) → drop-test results. Each is the median of its feasible set.",
      "First-shot feasibility 83–87% (19 masses × 6 conditions × 2 seeds). At real-waterbird conditions: generated 95%, measured bird leg 71%."),
  s8=("What the Generator Learned Without Being Told", "The power law is in the parameterisation; the independent evidence is segment ratios and posture.",
      "Segment ratios within 1% of the bird median; effective exponent 0.34 soft / 0.26 rigid, never at the box edge; ankle 17° lower."),
  s9=("One Real Swan, Identical Joints: What the Generator Changed", "Trumpeter swan, 11.07 kg, leg geometry entirely measured; the generator designs its own leg at the same mass.",
      "Joint stiffness is identical on both sides; only geometry and touchdown posture differ: L₁ 108 → 122 mm, ankle 144° → 128°. Peak 7.49 → 4.19 g (-44%), leg mass 127 → 93 g (-27%)."),
  s95=("Comparison with DJI Agras T30-Style Landing Gear",
       "Same material, same load case, comparable mass (709 g vs 669 g): the skid gives 4.7 mm of stroke and 33 g peak; A0 gives 22.6 mm and 7.9 g."),
  s10=("Summary", None,
       "Birds follow a tight leg-length law; inside it, flight efficiency trades against leg length, robust to phylogeny.\n"
       "Handed to a generator as a corridor, not a number: verified designs across the envelope at >80% first-shot feasibility.\n"
       "Where the generator departs from biology — rigid ground, posture — it departs for reasons we can name.",
       "Open: rebound criterion in acceptance · horizontal touchdown speed as input · hardware"))}[LANG]

# ---------- 打开源文件，只留架构页 ----------
prs = Presentation(SRC[0])
xml = prs.slides._sldIdLst
keep = None
for sid, sl in zip(list(xml), prs.slides):
    if any(sh.has_text_frame and SRC[1] in sh.text_frame.text for sh in sl.shapes):
        keep = sid; break
assert keep is not None, "没找到架构页"
for sid in list(xml):
    if sid is not keep:
        prs.part.drop_rel(sid.rId); xml.remove(sid)
ARCH = prs.slides[0]
# 保留页的 partname 会和新加页撞名（slide2.xml 重复 → 文件损坏），先改名
from pptx.opc.packuri import PackURI
ARCH.part.partname = PackURI("/ppt/slides/slide900.xml")
LAY = {l.name: l for l in prs.slide_masters[0].slide_layouts}
TITLE_LAY, BLANK_LAY = LAY["标题幻灯片"], LAY["空白"]

def txt(sl, x, y, w, h, s, size, bold, color, align=PP_ALIGN.LEFT, space=1.0):
    t = sl.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0); tf.margin_top = tf.margin_bottom = Emu(0)
    for i, ln in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln; p.alignment = align; p.line_spacing = space
        for r in p.runs:
            r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = YH
    return t
def page(title, _=None):
    """标题一行，不再有副标题条（副标题在学术场合是多余的）。"""
    sl = prs.slides.add_slide(BLANK_LAY)
    txt(sl, 0.401, 0.30, 11.2, 0.60, title, 26, True, CRIM)
    return sl
def bottom(sl, s, size=14):
    sh = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, I(0.652), I(6.605), I(12.03), I(0.717))
    sh.fill.solid(); sh.fill.fore_color.rgb = CRIM2; sh.line.fill.background(); sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = I(0.12)
    for i, ln in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln; p.alignment = PP_ALIGN.CENTER; p.line_spacing = 1.05
        for r in p.runs:
            r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = WHITE; r.font.name = YH
def fig(sl, path, box=(0.40, 1.62, 12.53, 4.86)):
    x, y, w, h = box; iw, ih = Image.open(path).size; ar = iw / ih
    ww, hh = (h * ar, h) if w / h > ar else (w, w / ar)
    sl.shapes.add_picture(path, I(x + (w - ww) / 2), I(y + (h - hh) / 2), I(ww), I(hh))


def rich(sl, x, y, w, h, parts, size=15.5, space=1.42):
    """一段正文，parts = [(文字, 0=普通/1=红色加粗), ...]，同一段内混排。"""
    t = sl.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0); tf.margin_top = tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]; p.line_spacing = space; p.alignment = PP_ALIGN.JUSTIFY
    for txt_, hl in parts:
        r = p.add_run(); r.text = txt_
        r.font.size = Pt(size); r.font.name = YH
        r.font.bold = bool(hl); r.font.color.rgb = CRIM if hl else INK
    return t

def bg_page(D):
    """课题背景：小标题 + 描述性正文 + 三张大图 + 三个标签 + 结论条。"""
    sl = page(D[0])
    sq = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, I(0.62), I(1.10), I(0.13), I(0.13))
    sq.fill.solid(); sq.fill.fore_color.rgb = CRIM2
    sq.line.fill.background(); sq.shadow.inherit = False
    txt(sl, 0.88, 0.99, 10.0, 0.42, D[1], 19, True, INK)
    rich(sl, 0.88, 1.55, 11.60, 1.88, D[2], 15.5 if LANG == "cn" else 13.0)
    for i, nm in enumerate(("BG1_skid", "BG2_load", "BG3_design")):
        fig(sl, P(nm), (0.88 + i * 3.91, 3.50, 3.78, 2.14))
    for i, lab in enumerate(D[3]):
        txt(sl, 0.88 + i * 3.91, 5.76, 3.78, 0.40, lab, 17, True, CRIM2, PP_ALIGN.CENTER)
    bottom(sl, D[4], 16)
    return sl

# 1 首页
sl = prs.slides.add_slide(TITLE_LAY)
for ph in list(sl.placeholders): ph._element.getparent().remove(ph._element)
txt(sl, 0.666, 2.30, 12.38, 1.50, T["t1"][0], 32 if LANG == "en" else 34, True, WHITE, space=1.15)
txt(sl, 0.666, 3.95, 12.38, 0.60, T["t1"][1], 17, False, WHITE)
txt(sl, 3.59, 5.23, 9.45, 1.10, T["t1"][2], 18, False, WHITE, space=1.30)

# 2 课题背景
bg_page(T["bg"])

# 2 架构页：改标题/副标题为学术风
for sh in ARCH.shapes:
    if sh.name == "TextBox 1" and sh.has_text_frame:
        r = sh.text_frame.paragraphs[0].runs[0]; r.text = T["arch"][0]
        r.font.size = Pt(26)
        for extra in sh.text_frame.paragraphs[0].runs[1:]: extra.text = ""
    if sh.name == "矩形 5":                       # 副标题条:整套 PPT 都不要
        sh._element.getparent().remove(sh._element)
    if sh.has_text_frame and "Data Processing" in sh.text_frame.text:
        for _p in sh.text_frame.paragraphs:
            for r in _p.runs:
                r.text = r.text.replace("Data Processing", "Data Engine")

# 3–8（A0 先行：先定义 9 个设计量，再讲先验从哪来）
sl = page(T["s6"][0]); fig(sl, P("F8_a0"), (0.40, 1.18, 12.53, 5.30)); bottom(sl, T["s6"][2])
sl = page(T["s3"][0]); fig(sl, P("F_prior"), (0.40, 1.20, 12.53, 5.28)); bottom(sl, T["s3"][2])
sl = page(T["s4"][0]); fig(sl, P("F_falcon"), (0.40, 1.18, 12.53, 5.30)); bottom(sl, T["s4"][2])
sl = page(T["s5"][0]); fig(sl, P("F46_replication"), (0.40, 1.18, 12.53, 5.30)); bottom(sl, T["s5"][2], 13.5)
sl = page(T["s7"][0]); fig(sl, P("F_generate"), (0.40, 1.15, 12.53, 5.34)); bottom(sl, T["s7"][2], 13.5)
sl = page(T["s8"][0]); fig(sl, P("F_agree"), (0.40, 1.18, 12.53, 5.30)); bottom(sl, T["s8"][2], 13.5)

# 9 落震（视频存在才插）
MP4, PNG = f"{U}/outputs/anim_swan/swan_vs_gen.mp4", f"{U}/outputs/anim_swan/swan_vs_gen_last.png"
HAS_VIDEO = os.path.exists(MP4) and os.path.exists(PNG)
if HAS_VIDEO:
    sl = page(T["s9"][0])
    iw, ih = Image.open(PNG).size; w = 11.9; h = w * ih / iw
    if h > 5.20: h = 5.20; w = h * iw / ih
    sl.shapes.add_movie(MP4, I(0.40 + (12.53 - w) / 2), I(1.25 + (5.20 - h) / 2), I(w), I(h),
                        poster_frame_image=PNG, mime_type="video/mp4")
    bottom(sl, T["s9"][2])
else:
    print("⚠ 无落震视频，跳过第 9 页")

# 10 与常规起落架对比
sl = page(T["s95"][0]); fig(sl, P("F_baseline"), (0.40, 1.18, 12.53, 5.30)); bottom(sl, T["s95"][1], 13.5)

# 11 总结
sl = page(T["s10"][0], None)
txt(sl, 0.9, 1.5, 11.5, 3.8, T["s10"][2], 20 if LANG == "cn" else 17, False, INK, space=1.55)
bottom(sl, T["s10"][3], 14)

# 架构页挪到第 3 位（首页 · 课题背景 · 架构）
ids = list(xml); xml.remove(keep); xml.insert(2, keep)   # 首页、背景页之后

# 备注
n = len(xml)
cn = SCRIPT_CN if HAS_VIDEO else SCRIPT_CN[:8] + SCRIPT_CN[9:]
en = SCRIPT if HAS_VIDEO else SCRIPT[:8] + SCRIPT[9:]
assert n == len(cn) == len(en), (n, len(cn), len(en))
RULE = "\n\n" + "—" * 34 + "\n"
for slide, c, e in zip(prs.slides, cn, en):
    slide.notes_slide.notes_text_frame.text = "【中文稿】\n\n" + c.strip() + RULE + "【English script】\n\n" + e.strip()

try: prs.save(OUT)
except PermissionError:
    OUT = OUT.replace(".pptx", "_new.pptx"); prs.save(OUT); print("⚠ 被占用，另存 _new")
print("→", OUT, n, "页")
