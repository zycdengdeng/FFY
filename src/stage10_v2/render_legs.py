# -*- coding: utf-8 -*-
"""真实设计的 3D 渲染:五个体重的推理结果,黑底,分段配色,静态多面板 + 旋转 GIF。
几何与仿真严格一致:着陆姿态 a1=50°(跖), thetaA=120°, thetaK=90°;
杆件画成真实外径的圆柱(结构定尺链输出),足端球 r=0.2·L1,机身球 ∝ m^(1/3)。"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

BG = "#0E1116"
COL = {"tarso": "#2EC4A0", "tibio": "#F2695C", "femur": "#5AA7E8"}
INK, MUT = "#E8EAED", "#9AA0A6"
plt.rcParams.update({"font.sans-serif": ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"],
                     "axes.unicode_minus": False})

D = json.load(open("/tmp/designs.json"))

def chain(l1, l2, l3):
    a1, a2, a3 = np.radians(50), np.radians(110), np.radians(20)
    d = lambda a: np.array([np.cos(a), 0., np.sin(a)])
    rf = 0.2 * l1
    F = np.array([0., 0., rf])
    A = F + l1 * d(a1); K = A + l2 * d(a2); H = K + l3 * d(a3)
    return F, A, K, H, rf

def cyl(ax, p0, p1, r, color, n=14):
    v = p1 - p0; L = np.linalg.norm(v); v = v / L
    a = np.array([0, 0, 1.]) if abs(v[2]) < 0.99 else np.array([1., 0, 0])
    n1 = np.cross(v, a); n1 /= np.linalg.norm(n1); n2 = np.cross(v, n1)
    t = np.linspace(0, L, 2); th = np.linspace(0, 2*np.pi, n)
    t, th = np.meshgrid(t, th)
    P = (p0[:, None, None] + v[:, None, None]*t
         + r*(n1[:, None, None]*np.cos(th) + n2[:, None, None]*np.sin(th)))
    ax.plot_surface(P[0], P[1], P[2], color=color, shade=True,
                    lightsource=matplotlib.colors.LightSource(azdeg=200, altdeg=55),
                    linewidth=0, antialiased=True, alpha=1.0)

def sphere(ax, c, r, color, alpha=1.0, n=16):
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, n), np.linspace(0, np.pi, n))
    ax.plot_surface(c[0]+r*np.cos(u)*np.sin(v), c[1]+r*np.sin(u)*np.sin(v),
                    c[2]+r*np.cos(v), color=color, shade=True, linewidth=0, alpha=alpha)

def draw_leg(ax, d, x0=0.0):
    l1, l2, l3 = d["L_mm"]; D1, D2, D3 = d["D_mm"]
    F, A, K, H, rf = chain(l1, l2, l3)
    off = np.array([x0, 0, 0])
    sphere(ax, F+off, rf, "#5B6470")                       # 足端(蹼足等效球)
    cyl(ax, F+off, A+off, D1/2, COL["tarso"])
    cyl(ax, A+off, K+off, D2/2, COL["tibio"])
    cyl(ax, K+off, H+off, D3/2, COL["femur"])
    for J, r in ((A, max(D1, D2)*0.75), (K, max(D2, D3)*0.75)):
        sphere(ax, J+off, r, "#C8CDD3")                    # 关节
    rb = 28 * d["m"] ** (1/3)                              # 机身按体积∝m
    sphere(ax, H+off+np.array([0, 0, rb*0.75]), rb, "#3A4250", alpha=.92)
    return H, rb

def style(ax, lim, zlim):
    ax.set_facecolor(BG)
    ax.set_xlim(*lim); ax.set_ylim(-(lim[1]-lim[0])/2, (lim[1]-lim[0])/2); ax.set_zlim(*zlim)
    ax.set_box_aspect((lim[1]-lim[0], lim[1]-lim[0], zlim[1]-zlim[0]), zoom=1.55)
    ax.set_axis_off()

# ---------------- 静态多面板 ----------------
fig = plt.figure(figsize=(13, 4.4), facecolor=BG)
for i, d in enumerate(D):
    ax = fig.add_subplot(1, 5, i+1, projection="3d", facecolor=BG)
    H, rb = draw_leg(ax, d)
    style(ax, (-75, 215), (0, 425))
    ax.view_init(elev=10, azim=-68)
    ax.set_title(f"m = {d['m']:g} kg", color=INK, fontsize=13, pad=-26)
    txt = (f"L₁/L₂/L₃ = {d['L_mm'][0]:.0f}/{d['L_mm'][1]:.0f}/{d['L_mm'][2]:.0f} mm\n"
           f"管径 {d['D_mm'][0]:.1f}/{d['D_mm'][1]:.1f}/{d['D_mm'][2]:.1f} mm\n"
           f"峰值 {d['peak_g']:.1f} g · 腿重 {d['leg_mass_g']:.0f} g")
    ax.text2D(0.5, -0.045, txt, transform=ax.transAxes, color=MUT,
              fontsize=7.8, ha="center", va="top", linespacing=1.5)
fig.suptitle("条件生成 → 真值仿真选优 → 结构定尺:五个体重下的真实推理结果(同尺渲染)",
             color=INK, fontsize=13, y=0.99)
fig.text(0.40, 0.925, "■ 跖 L₁", color=COL["tarso"], fontsize=11)
fig.text(0.475, 0.925, "■ 胫 L₂", color=COL["tibio"], fontsize=11)
fig.text(0.55, 0.925, "■ 股 L₃", color=COL["femur"], fontsize=11)
fig.savefig("/tmp/legs_grid.png", dpi=220, bbox_inches="tight",
            facecolor=BG, pad_inches=0.25)
print("grid saved")

# ---------------- 旋转 GIF:每腿绕自身立轴自转,镜头固定 ----------------
def chain_rot(l1, l2, l3, phi):
    """腿绕过足端的竖直轴旋转 phi(原地自转)。"""
    a1, a2, a3 = np.radians(50), np.radians(110), np.radians(20)
    d = lambda a: np.array([np.cos(a)*np.cos(phi), np.cos(a)*np.sin(phi), np.sin(a)])
    rf = 0.2 * l1
    F = np.array([0., 0., rf])
    A = F + l1*d(a1); K = A + l2*d(a2); H = K + l3*d(a3)
    return F, A, K, H, rf

def draw_leg_rot(ax, d, phi, x0):
    l1, l2, l3 = d["L_mm"]; D1, D2, D3 = d["D_mm"]
    F, A, K, H, rf = chain_rot(l1, l2, l3, phi)
    off = np.array([x0, 0, 0])
    sphere(ax, F+off, rf, "#5B6470")
    cyl(ax, F+off, A+off, D1/2, COL["tarso"])
    cyl(ax, A+off, K+off, D2/2, COL["tibio"])
    cyl(ax, K+off, H+off, D3/2, COL["femur"])
    for J, r in ((A, max(D1, D2)*0.75), (K, max(D2, D3)*0.75)):
        sphere(ax, J+off, r, "#C8CDD3")
    rb = 28 * d["m"] ** (1/3)
    sphere(ax, H+off+np.array([0, 0, rb*0.75]), rb, "#3A4250", alpha=.92)

GAP = 205
XS = [i*GAP for i in range(len(D))]
fig2 = plt.figure(figsize=(11, 4.6), facecolor=BG)
ax = fig2.add_subplot(111, projection="3d", facecolor=BG)
ax.set_position([-0.24, -0.30, 1.48, 1.58])   # 画轴撑出画布,吃掉 3D 默认白边

def frame(k):
    ax.clear(); ax.set_facecolor(BG)
    phi = np.radians(k*5)
    for d, xx in zip(D, XS):
        draw_leg_rot(ax, d, phi, xx)
        ax.text(xx, 0, -60, f"{d['m']:g} kg", color=INK, fontsize=11, ha="center")
    ax.set_xlim(-140, XS[-1]+150); ax.set_ylim(-150, 150); ax.set_zlim(-40, 430)
    ax.set_box_aspect((XS[-1]+290, 380, 470), zoom=1.0)
    ax.set_axis_off()
    ax.view_init(elev=11, azim=-88)
    return []

fig2.text(0.5, 0.94, "条件生成的真实起落架设计 · 1 → 12 kg", color=INK,
          fontsize=13, ha="center")
fig2.text(0.5, 0.885, "同尺渲染 · 杆件为结构定尺输出的真实管径 · 跖L₁ 胫L₂ 股L₃",
          color=MUT, fontsize=9.5, ha="center")
ani = FuncAnimation(fig2, frame, frames=72, blit=False)
ani.save("/tmp/legs_rotate.gif", writer=PillowWriter(fps=18),
         savefig_kwargs=dict(facecolor=BG))
print("gif saved")
