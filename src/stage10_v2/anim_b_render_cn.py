# -*- coding: utf-8 -*-
"""中文版并排落震视频：复用英文渲染脚本的全部布局，只换字符串与字体。
用法: python src/stage10_v2/anim_b_render_cn.py outputs/anim_b [slow=150]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "anim_b_render_en.py"), encoding="utf-8").read()
M = [
 ('plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,',
  'plt.rcParams.update({'),
 ('                     "figure.facecolor": "white", "savefig.facecolor": "white"})',
  '                     "figure.facecolor": "white", "savefig.facecolor": "white"})\n'
  'import matplotlib.font_manager as _fm\n'
  '_have={f.name for f in _fm.fontManager.ttflist}\n'
  '_cjk=next((f for f in ("Noto Sans CJK SC","Noto Sans CJK JP","WenQuanYi Zen Hei","Droid Sans Fallback")\n'
  '           if f in _have), None)\n'
  'plt.rcParams["font.sans-serif"]=([_cjk] if _cjk else [])+["DejaVu Sans"]\n'
  'plt.rcParams["font.family"]="sans-serif"\n'
  'plt.rcParams["axes.unicode_minus"]=False'),
 ('"BIRD\'S EXPONENT"', '"鸟的指数"'),
 ('"MACHINE\'S EXPONENT"', '"机器的指数"'),
 ("mm       leg mass {m['leg_mass_g']:.0f} g", "mm       腿重 {m['leg_mass_g']:.0f} g"),
 ('"deceleration / g", 10.0, "10 g limit"', '"峰值过载 / g", 10.0, "10 g 上限"'),
 ('"stroke / mm", 24.0, "24 mm budget"', '"落震行程 / mm", 24.0, "24 mm 预算"'),
 ('"time after touchdown / ms"', '"触地后时间 / ms"'),
 ('label=LAB[k][0].title()', 'label=LAB[k][0]'),
 ('f"Same joints, same {J[\'m_kg\']:.0f} kg airframe, same {J[\'v0\']} m/s touchdown "',
  'f"同一套关节、同一个 {J[\'m_kg\']:.0f} kg 机身、同一个 {J[\'v0\']} m/s 触地速度 "'),
 ('f"— only the leg-length scaling law differs      ({SLOW:g}$\\\\times$ slow motion)"',
  'f"—— 只有腿长标度律不同      （{SLOW:g}× 慢放）"'),
 ('f"peak  {b[\'peak_g\']:.2f} g  vs  {p[\'peak_g\']:.2f} g  ({100*(p[\'peak_g\']/b[\'peak_g\']-1):+.0f}%)"',
  'f"峰值 {b[\'peak_g\']:.2f} g  对  {p[\'peak_g\']:.2f} g（{100*(p[\'peak_g\']/b[\'peak_g\']-1):+.0f}%）"'),
 ('f"      ·      stroke  {b[\'leg_stroke_mm\']:.1f} mm  vs  {p[\'leg_stroke_mm\']:.1f} mm"',
  'f"      ·      行程 {b[\'leg_stroke_mm\']:.1f} mm  对  {p[\'leg_stroke_mm\']:.1f} mm"'),
 ('f"      ·      leg mass  {b[\'leg_mass_g\']:.0f} g  vs  {p[\'leg_mass_g\']:.0f} g  "',
  'f"      ·      腿重 {b[\'leg_mass_g\']:.0f} g  对  {p[\'leg_mass_g\']:.0f} g "'),
 ('f"({100*(p[\'leg_mass_g\']/b[\'leg_mass_g\']-1):+.0f}%)   — both feasible"',
  'f"（{100*(p[\'leg_mass_g\']/b[\'leg_mass_g\']-1):+.0f}%）   —— 两者都可行"'),
 ('f"b_compare_{TAG}_en_last.png"', 'f"b_compare_{TAG}_cn_last.png"'),
 ('f"b_compare_{TAG}_en.mp4"', 'f"b_compare_{TAG}_cn.mp4"'),
]
for a, b in M:
    assert a in src, "未匹配: " + a[:60]
    src = src.replace(a, b)
exec(compile(src, "anim_b_render_cn", "exec"), {"__name__": "__main__", "__file__": __file__})
