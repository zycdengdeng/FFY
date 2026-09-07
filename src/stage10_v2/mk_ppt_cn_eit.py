# -*- coding: utf-8 -*-
"""本周汇报 PPT 中文版（13 页，EIT 模板）—— 供审阅/修改，改定后再同步回英文版。
与英文版的差别：全部文字中文化；多出「隼科具体例子」一页（第 6 页）。"""
import os
from pptx import Presentation
from pptx.util import Inches as I, Pt, Emu
from pptx.dml.color import RGBColor as C
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

U = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F, AN = f"{U}/outputs/ppt_cn", f"{U}/outputs/anim_b"
TPL = f"{U}/每周汇报/09_09_2026_Week5_International_Meeting_EN.pptx"
OUT = f"{U}/每周汇报/07_09_2026_Week7_Group_Meeting_CN.pptx"
CRIM, CRIM2, WHITE = C(0x8E, 0x2A, 0x34), C(0x7F, 0x2D, 0x32), C(0xFF, 0xFF, 0xFF)
INK = C(0x1A, 0x1A, 0x1A)
YH = "微软雅黑"

prs = Presentation(TPL)
sl_lst = prs.slides._sldIdLst
for sid in list(sl_lst):
    prs.part.drop_rel(sid.rId); sl_lst.remove(sid)
LAY = {l.name: l for l in prs.slide_masters[0].slide_layouts}
TITLE_LAY, BLANK_LAY = LAY["标题幻灯片"], LAY["空白"]

def txt(sl, x, y, w, h, s, size, bold, color, align=PP_ALIGN.LEFT, space=1.0):
    t = sl.shapes.add_textbox(I(x), I(y), I(w), I(h)); tf = t.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0); tf.margin_top = tf.margin_bottom = Emu(0)
    for i, ln in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln; p.alignment = align; p.line_spacing = space
        for r in p.runs:
            r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = YH
    return t

def page(title, kicker):
    sl = prs.slides.add_slide(BLANK_LAY)
    txt(sl, 0.401, 0.254, 9.85, 0.505, title, 24, True, CRIM)
    if kicker: txt(sl, 0.401, 0.900, 12.18, 0.60, kicker, 15, True, CRIM2, space=1.12)
    return sl

def bottom(sl, s, size=14):
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

BOX = (0.40, 1.62, 12.53, 4.86)
def fig(sl, path, box=BOX):
    x, y, w, h = box
    iw, ih = Image.open(path).size; ar = iw/ih
    ww, hh = (h*ar, h) if w/h > ar else (w, w/ar)
    sl.shapes.add_picture(path, I(x+(w-ww)/2), I(y+(h-hh)/2), I(ww), I(hh))

def notes(sl, s): sl.notes_slide.notes_text_frame.text = s

# ============ 1 首页 ============
sl = prs.slides.add_slide(TITLE_LAY)
for ph in list(sl.placeholders): ph._element.getparent().remove(ph._element)
txt(sl, 0.666, 2.35, 12.38, 1.50, "8,705 只鸟能告诉我们起落架该多长吗？", 34, True, WHITE, space=1.15)
txt(sl, 0.666, 3.95, 12.38, 0.55,
    "一条做过亲缘校正的腿长定律 —— 以及机器在哪一点上不同意它", 18, False, WHITE)
txt(sl, 3.59, 5.23, 9.45, 1.10,
    "第七周 · 上海交大–宁波东方理工联培 · 王子晗 · 导师：赵一帆、夏焜\n2026 年 9 月 7 日",
    18, False, WHITE, space=1.30)
notes(sl, "15 分钟。这次讲「我们在做什么」，不是「我们做了多少」。"
          "两半：(1) 从鸟身上抽出一条定律，以及我们怎么想尽办法去推翻它；"
          "(2) 把这条定律交给机器之后，机器在哪一个数上不同意生物。")

# ============ 2 数据集 ============
sl = page("我们搭的数据集", "三个公开数据源，一次拼接 —— 而且每一行都挂上了它在系统树上的位置。")
fig(sl, f"{F}/cn_F7_dataset.png")
bottom(sl, "8,705 个物种 · 188 个科 · 35 个目，每一行同时带着形态、生态、和亲缘位置。")
notes(sl, "AVONET 给形态，EltonTraits 给生态，BirdTree 给亲缘。BirdLife 与 BirdTree 两套命名法不同，"
          "需要做 crosswalk，98.6% 对上。不会飞的类群剔除——鸵鸟会用错误的理由主导任何腿长回归。"
          "HWI = Kipp 距离 / 翼长，是这个文献里飞行效率的标准代理量：翼尖越尖、展弦比越高，"
          "长距离飞行越省力。")

# ============ 3 定律一 ============
sl = page("第一步 · 先拟合显而易见的定律，然后把它扔掉",
          "经典异速生长。我们真正要留下的，是这条定律解释不了的那一部分：残差 u。")
fig(sl, f"{F}/cn_F1_allometry.png")
bottom(sl, "设计先验 = 做过亲缘校正的水鸟拟合：b = 0.365，σ = 0.081 dex。这一页之后所有内容都是关于 u 的。")
notes(sl, "左：8,705 种。橙色虚线是不管亲缘直接回归；绿线是我们真正取先验的类群——水鸟，"
          "生态上最接近一台落在软而不平地面上的重型 VTOL——并且把亲缘协方差放进误差项拟合。"
          "b=0.365 表示腿长比体重长得慢；几何相似应当是 1/3，所以略高于等比。"
          "右：减掉这条线、再除以散度，得到 u，即「同体重下腿长的 z 分数」。"
          "u<0 = 腿比该体重应有的更短。后面所有生物结论的因变量都是 u。")

# ============ 4 设计先验长什么样 ============
sl = page("先验到底长什么样",
          "把定律一落到我们真正要设计的那段体重上：一条中线 + 一条 ±2.5σ 的走廊。")
fig(sl, f"{F}/cn_F_prior.png", (0.40, 1.62, 12.53, 4.86))
bottom(sl, "这条走廊就是交给生成器的东西：给定目标质量，给的是一段合理腿长，而不是一个数。"
           "12 kg 以上没有水鸟数据，属外推。")
notes(sl, "这是我们真正交给生成器的先验，画在原始尺度上。红线是水鸟拟合 b=0.366，"
          "外面套一条 ±2.5σ 的走廊。水鸟是我们盯得最紧的一支——它们低速降落在软而不平的地面上，"
          "而且我们的感知管线取的也是水鸟的素材，这个我单独讲。"
          "两条灰线是理论参考：几何相似 b=1/3、弹性相似 b=1/4，鸟比这两条都陡，"
          "说明腿长比纯几何缩放长得更快。绿色是产品区间 5–30 kg。"
          "要老实说的是 12 kg 以上没有水鸟——天鹅鹈鹕就到那儿了，我们区间的上半段是外推，"
          "这也正是重的那一头必须靠仿真而不是靠鸟来定的原因。")

# ============ 5 定律二 ============
sl = page("第二步 · 一个生态变量挣到了进入公式的资格",
          "HWI = 手翼指数 = Kipp 距离 / 翼长。HWI 高 = 翼尖尖、展弦比高 = 长距离飞行省力。")
fig(sl, f"{F}/cn_F2_hwi.png")
bottom(sl, "只多一项，R² 从 0.671 提到 0.793。从麻雀到水鸟这么大的翼型差别，预测同体重下腿长少 31%。")
notes(sl, "这是我们之前没做过的一版拟合：把 HWI 直接放进腿长公式，而不是只对残差 u 做回归。"
          "多这一项，R² 从 0.67 提到 0.79——吸收掉纯体重定律留下的 37% 方差。系数 −0.0061/HWI，t=−71。"
          "可以直接读成一个权衡：为高效飞行投资的鸟，要拿腿来付账。"
          "这正是起落架设计者面对的同一笔交易——行程 vs 带着走的质量。")

# ============ 5 PGLS ============
sl = page("第三步 · 生物学家第一个会提的反驳",
          "亲缘接近的物种不是独立数据点。系统发育广义最小二乘，200 棵树 × 2 个骨架。")
fig(sl, f"{F}/cn_F3_pgls.png", (0.40, 1.68, 12.53, 4.70))
bottom(sl, "共同祖先吃掉了朴素效应的 65% —— 剩下的 35% 在每一棵树上都活下来了（p < 1e-56，λ = 0.93）。")
notes(sl, "这页是给生物学家看的。普通最小二乘假设残差独立，但在系统树上并不独立——两只雨燕像，"
          "是因为共同祖先，不是因为各自独立解出了同一个答案。PGLS 把树带来的期望协方差放进误差项。"
          "λ=0.93 说明这个性状几乎完全被亲缘结构化，所以修正很狠：系数从 −0.078 掉到 −0.027。"
          "但 200 棵树里没有一棵让它碰到 0。我们报的是缩水后站得住的那个值，不是好看的那个。"
          "补充：新跑完的协变量-OU（T1）在 20 棵树上独立复算出同一个 −0.0272，"
          "而且在 BM/λ/OU 三种演化过程下，加飞行效率的 ΔAIC 都在 1200 以上。")

# ============ 6 隼科具体例子 ============
sl = page("这条权衡长什么样 —— 隼科的两只隼",
          "在讲统计之前，先看一个最容易理解的情形：同一个科、亲缘极近、都吃肉、都会飞。")
fig(sl, f"{F}/cn_F_falcon.png", (0.40, 1.58, 12.53, 4.92))
bottom(sl, "林隼比烟隼还轻 65 克，腿却长了 81%。整个隼科 62 种，飞行效率与腿长残差的相关是 r = −0.68。")
notes(sl, "这一页是全场最容易懂的一页，讲慢一点。烟隼是开阔天空的长距离迁徙猎手，高速平飞追捕，"
          "腿对它是纯粹的死重和阻力，跗跖只有 34 mm。林隼在南美雨林树冠层里穿梭，靠短距爆发追捕，"
          "还要在枝上站立抓握，跗跖 62 mm——而它比烟隼还轻 65 克。"
          "关键是：这两个物种同属隼科，祖先几乎一样、食性一样、都会飞，"
          "所以差别不能推给「祖先不同」或「生活方式差太远」，只能来自飞行方式本身。"
          "右边是整个科 62 个物种，趋势一致，r=−0.68。按 HWI=47 把科内切两半，"
          "两半的中位残差差 −1.48——这就是下一页那张图里隼科的那一根柱子。")

# ============ 8 科内重复 ============
sl = page("第四步之一 · 让这棵树自己跟自己吵架：科内部",
          "如果这只是一次古老的偶然，它不需要在每一个科的内部都重演一遍。")
fig(sl, f"{F}/cn_F4_within.png", (0.40, 1.58, 12.53, 4.92))
bottom(sl, "96 个科里 77 个在内部重演（随机应为 50%，z = 5.9）；收紧到属，419 个里仍有 251 个（60%，z = 4.1）。")
notes(sl, "就是上一页隼科那件事，重复 96 遍。在每个科内部先扣掉体重，再按飞行效率分成高低两半来比。"
          "祖先相同、生活方式相似，剩下能解释差别的只有飞行效率。80% 的科朝预期方向走。"
          "右下角那行是更严的版本：连属都不许跨，419 个属里 251 个仍然朝同一个方向。")

# ============ 9 跨支系跳变 ============
sl = page("第四步之二 · 让这棵树自己跟自己吵架：跨支系",
          "换一个尺度：不看科内部，看整棵树上哪些支系在自己的干上把腿改掉了。")
fig(sl, f"{F}/cn_F_shifts.png", (0.40, 1.58, 12.53, 4.92))
bottom(sl, "24 次互不嵌套的独立跳变：飞得好的 12 支里 10 支缩短，飞得差的 12 支里只有 6 支；"
           "跳变幅度与该支系飞行效率的相关 r = −0.59。")
notes(sl, "每一根柱子是一个支系在自己的干上发生的腿长跳变 Δu，灰线是该支系的平均飞行效率。"
          "这里只留互不嵌套的 24 个——一开始我数了 40 个，但其中 16 个嵌套在别的支系里，"
          "同一次演化事件被数了两遍；去掉重复计数之后相关反而从 −0.40 变强到 −0.59。"
          "左半（飞得好的）以蓝色缩短为主，右半（飞得差的）红色伸长明显变多。"
          "个别例外要讲清楚：䴙䴘科和鹈形目都是 +（腿变长），它们是潜水和涉水的类群，"
          "腿在水里另有用途——这正好是下一页觅食层位那条证据要讲的事。")

# ============ 10 功能检验 ============
sl = page("第五步 · 换一个完全独立的数据集做功能检验",
          "如果这条定律是真的，腿长应该也跟着腿的用途走，而不只是跟着翼型走。")
fig(sl, f"{F}/cn_F5_forstrat.png", (0.40, 1.68, 12.53, 4.70))
bottom(sl, "空中取食的腿最短（−2.6 标准差），地面取食的最长（+0.9）；偏相关的正负号"
           "恰好在功能该翻的地方翻了过来。")
notes(sl, "EltonTraits 给出每个物种在各个觅食层位的百分比。在空中取食的鸟几乎不用腿，腿就极短；"
          "在地面行走觅食的鸟需要腿，也要为它付钱。控制体重后，地面觅食比例与腿长残差偏相关 +0.29，"
          "树上 −0.21。水域最弱，只有 −0.06，因为水鸟本身很不均质——涉水的鹭和潜水的鸬鹚都算「水域」。"
          "这是这一页诚实的保留项。")

# ============ 11 管线 ============
sl = page("把这条定律交给机器", "先验并不挑设计。它挑的是：生成器被允许在哪一片区域里搜。")
fig(sl, f"{F}/cn_F9_pipeline.png", (0.40, 1.72, 12.53, 4.62))
bottom(sl, "给的是一条 1σ 的走廊，不是一个数 —— 这样物理还有机会对生成器说「不行」。")
notes(sl, "对力学背景的听众，这一页的定位很重要：我们不是在抄一只鸟。定律给的是「目标质量下"
          "合理腿长的走廊」；生成器在 5 维工况条件下提出 9 维设计；每一个候选都由完整多体落震仿真"
          "校核，而不是由代理模型打分。生物负责收窄搜索，物理负责拍板。")

# ============ 12 A0 ============
sl = page("工程部分只讲一页 —— A0 腿",
          "三段、两套减震器、膝为自由铰 —— 这个选型是制造性决定的，不是性能决定的。")
fig(sl, f"{F}/cn_F8_a0.png", (0.40, 1.58, 12.53, 4.92))
bottom(sl, "架构消融显示三个候选的差异都落在换随机种子造成的噪声地板以内 —— "
           "所以 A0 赢在零件最少、以及弹簧能直接买到。")
notes(sl, "三段的命名沿用鸟：跗跖骨、胫跗骨、股骨。两套减震器而不是三套：我们消融掉膝簧，"
          "峰值从 3.60 g 变到 3.52 g，低于换随机种子就能造成的 2.9%。既然性能分不出架构，"
          "那就由制造性来分：A0 在股骨上保住 36 mm 力臂，把 1385 kN/m 的髋簧变成 616 kN/m——"
          "这是「特殊订制」和「货架件」的区别。")

# ============ 13 机器自己走出这条律（核心页）============
sl = page("机器自己走出了这条律",
          "生成器从没被告诉过「腿长要随体重按幂律走」——它对每个体重独立求解，结果自己走出了这条律。")
fig(sl, f"{F}/cn_F_agree.png", (0.40, 1.58, 12.53, 4.92))
bottom(sl, "定律的形式一致（R² = 0.98–0.99）；在鸟真正着陆的软地面上，指数也几乎一样："
           "机器 0.34–0.35 对鸟 0.366。只有到刚性地面才变平到 0.263。")
notes(sl, "占位——讲稿见文件末尾的 SCRIPT。")

# ============ 14 一次落震（视频，按新口径重跑后才插入）============
import os as _os
_MP4 = f"{AN}/b_compare_hard_en.mp4"; _PNG = f"{AN}/b_compare_hard_en_last.png"  # 视频统一用英文版
if _os.path.exists(_MP4) and _os.path.exists(_PNG):
    sl = page("这点差别，换算成一次落震是多少",
              "同一个 30 kg 机身、同一个触地速度、刚性地面、同一套关节。只有标度指数不同：0.366 对 0.263。")
    _iw, _ih = Image.open(_PNG).size
    _w = 12.0; _h = _w*_ih/_iw
    if _h > 4.86: _h = 4.86; _w = _h*_iw/_ih
    sl.shapes.add_movie(_MP4, I(0.40+(12.53-_w)/2), I(1.62+(4.86-_h)/2), I(_w), I(_h),
                        poster_frame_image=_PNG, mime_type="video/mp4")
    bottom(sl, "下一步：把回弹判据补进可行性检验 · 水平触地速度进条件向量 · 硬件。")
    notes(sl, "占位——讲稿见文件末尾的 SCRIPT。")
else:
    print("⚠ 未找到按新口径重跑的落震视频，已跳过第 14 页")

# ---- 在每页原有中文说明后面，追加同一份英文讲稿（改 speaker_script_en.py 即可）----
from speaker_script_en import SCRIPT  # noqa: E402
_n = len(prs.slides._sldIdLst)
assert _n in (len(SCRIPT), len(SCRIPT)-1), f"讲稿 {len(SCRIPT)} 段,幻灯片 {_n} 页"
for _sl, _txt in zip(prs.slides, SCRIPT[:_n]):
    _tf = _sl.notes_slide.notes_text_frame
    _tf.text = (_tf.text.strip() + "\n\n" + "—" * 34 + "\n【英文讲稿】\n\n" + _txt.strip())

try:                                   # 文件可能正被 PowerPoint 打开
    prs.save(OUT)
except PermissionError:
    OUT = OUT.replace(".pptx", "_new.pptx")
    prs.save(OUT)
    print("⚠ 原文件被占用(多半开着 PowerPoint),已另存为 _new.pptx")
print("→", OUT, os.path.getsize(OUT)//1024, "KB,", len(prs.slides._sldIdLst), "页")
