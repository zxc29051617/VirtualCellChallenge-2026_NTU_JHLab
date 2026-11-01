"""Result figures used in general_model/README.md (2026-10-24). All numbers are from
pertrelay/model/SUMMARY.md and the mock-exam / benchmark result files.

    <python with matplotlib> docs/figs/make_figures.py docs/figs        (from general_model/)
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"; os.makedirs(OUT, exist_ok=True)
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(f): font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Noto Sans CJK TC", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8D99AE", "axes.labelcolor": "#2B2D42", "xtick.color": "#2B2D42", "ytick.color": "#2B2D42",
                     "axes.titleweight": "bold", "axes.titlesize": 17, "axes.titlecolor": "#1F3A5F", "savefig.dpi": 220})
NAVY, TEAL, ORANGE, GRAY, LIGHT = "#1F3A5F", "#2A9D8F", "#E76F51", "#8D99AE", "#D9E2EC"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white"); plt.close(fig)


def bar_labels(ax, bars, fmt="{:.3f}", dy=0.0, size=14, color=NAVY):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=size, color=color, fontweight="bold")


# 1. hidden-cell-type benchmark ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
names = ["各來源平均\n（原方法）", "AdaptNet", "AdaptNet\n＋RPE1 資料"]
cols = [GRAY, "#5E8BC0", NAVY]
b = axes[0].bar(names, [0.111, 0.131, 0.152], color=cols, width=0.62); bar_labels(axes[0], b, dy=0.003)
axes[0].set_ylim(0, 0.18); axes[0].set_title("方向相似度（越高越好，1 = 完全一致）"); axes[0].set_yticks([0, 0.05, 0.10, 0.15])
b = axes[1].bar(names, [0.738, 0.762, 0.766], color=cols, width=0.62); bar_labels(axes[1], b, dy=0.004)
axes[1].set_ylim(0.45, 0.8); axes[1].axhline(0.5, color=ORANGE, ls="--", lw=1.5)
axes[1].set_title("區分度（0.5 = 亂猜）")
fig.tight_layout(w_pad=3); save(fig, "fig_benchmark.png")

# 2. mock exam progress --------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.4))
labels = ["原方法\n（v1）", "訓練模型\n預測總量（v2）", "＋RPE1\n資料（v3）", "設定調整＋\n3 模型平均（v4）", "移除 5 個\n雜訊方向（v5）"]
vals = [0.272, 0.291, 0.293, 0.305, 0.306]
b = ax.bar(labels, vals, color=[GRAY, "#8FA6C2", "#6D8BB2", "#3F6FA6", NAVY], width=0.62); bar_labels(ax, b, dy=0.0008)
ax.set_ylim(0.25, 0.315); ax.set_ylabel("H1 模擬考總分"); ax.set_yticks([0.25, 0.27, 0.29, 0.31])
ax.text(-0.45, 0.3135, "註：縱軸從 0.25 開始", fontsize=12, color=GRAY)
fig.tight_layout(); save(fig, "fig_mock_progress.png")

# 3. ablation waterfall --------------------------------------------------------------------------------------
steps = [("原方法", 0.2723, None), ("改用 AdaptNet\n預測總量", 0.2910, "+0.019"), ("＋RPE1\n資料", 0.2934, "+0.002"),
         ("不移除\n對照組方向", 0.2987, "+0.005"), ("目標權重", 0.3020, "+0.003"), ("3 模型平均", 0.3055, "+0.004"),
         ("只移除\n5 個方向", 0.3063, "+0.001")]
fig, ax = plt.subplots(figsize=(11, 4.6))
prev = None
for i, (lab, v, d) in enumerate(steps):
    if prev is None:
        ax.bar(i, v - 0.26, bottom=0.26, color=GRAY, width=0.6); ax.text(i, v + 0.0006, f"{v:.3f}", ha="center", fontsize=13, color=NAVY, fontweight="bold")
    else:
        ax.bar(i, v - prev, bottom=prev, color=TEAL if i < len(steps) - 1 else NAVY, width=0.6)
        ax.plot([i - 1 + 0.3, i - 0.3], [prev, prev], color=GRAY, lw=1, ls=":")
        ax.text(i, v + 0.0006, d, ha="center", fontsize=13, color=NAVY, fontweight="bold")
    prev = v
ax.annotate("總分 0.306", xy=(len(steps) - 1, 0.3053), xytext=(len(steps) - 1, 0.2985), ha="center", fontsize=14, color=NAVY, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1))
ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[0] for s in steps], fontsize=13)
ax.set_ylim(0.26, 0.315); ax.set_ylabel("H1 模擬考總分"); ax.text(-0.4, 0.3135, "註：縱軸從 0.26 開始", fontsize=12, color=GRAY)
fig.tight_layout(); save(fig, "fig_ablation.png")

# 4. sign accuracy per hidden cell type -----------------------------------------------------------------------
folds = ["HepG2", "Jurkat", "H1", "K562", "HCT116\nHEK293T", "iPSC", "CD4 T"]
acc = [0.735, 0.716, 0.626, 0.609, 0.560, 0.558, 0.541]
colors = [TEAL, TEAL, GRAY, GRAY, GRAY, GRAY, GRAY]
fig, ax = plt.subplots(figsize=(10, 4.3))
b = ax.bar(folds, acc, color=colors, width=0.62); bar_labels(ax, b, fmt="{:.0%}", dy=0.006)
ax.axhline(0.5, color=ORANGE, ls="--", lw=1.5)
ax.set_ylim(0.4, 0.8); ax.set_ylabel("最有把握的 10 個基因\n方向猜對率")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(color=TEAL, label="必需基因實驗"), Patch(color=GRAY, label="以非必需基因為主"), Line2D([0], [0], color=ORANGE, ls="--", lw=1.5, label="丟銅板 = 50%")], frameon=False, loc="upper right", fontsize=13)
fig.tight_layout(); save(fig, "fig_sign_accuracy.png")

# 5. what-if: true signs ---------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.3))
x = [0, 1]; w = 0.34
b1 = ax.bar([i - w / 2 for i in x], [0.547, 0.044], w, color=GRAY, label="實際預測")
b2 = ax.bar([i + w / 2 for i in x], [1.173, 0.599], w, color=TEAL, label="假設方向全對")
bar_labels(ax, b1, fmt="{:.2f}", dy=0.02); bar_labels(ax, b2, fmt="{:.2f}", dy=0.02)
ax.set_xticks(x); ax.set_xticklabels(["方向正確（fid）", "方向排序（reach）"]); ax.set_ylabel("換算分數（1 = 實驗重複水準）")
ax.set_ylim(0, 1.35); ax.legend(frameon=False, fontsize=13, loc="upper right")
fig.tight_layout(); save(fig, "fig_true_signs.png")

# 6. training data --------------------------------------------------------------------------------------------
ctx = [("KOLF（iPSC）", 11364, "幹細胞類"), ("VIPK562", 6168, "血球類"), ("FengNF（iPSC）", 2873, "幹細胞類"), ("Jurkat", 2270, "血球類"),
       ("RPE1", 2204, "上皮類"), ("HepG2", 2160, "上皮類"), ("K562ess", 2021, "血球類"), ("FengF（iPSC）", 1885, "幹細胞類"),
       ("HEK293T", 1697, "上皮類"), ("HCT116", 1690, "上皮類"), ("K562", 1649, "血球類"), ("CD4 刺激 8 小時", 677, "血球類"),
       ("CD4 刺激 48 小時", 675, "血球類"), ("CD4 休息", 670, "血球類"), ("H1（胚胎幹細胞）", 300, "幹細胞類")]
cmap = {"血球類": TEAL, "上皮類": "#E9C46A", "幹細胞類": "#B5838D"}
fig, ax = plt.subplots(figsize=(8, 6.2))
ys = list(range(len(ctx)))[::-1]
ax.barh(ys, [c[1] for c in ctx], color=[cmap[c[2]] for c in ctx], height=0.68)
for y, c in zip(ys, ctx): ax.text(c[1] + 120, y, f"{c[1]:,}", va="center", fontsize=12, color="#2B2D42")
ax.set_yticks(ys); ax.set_yticklabels([c[0] for c in ctx], fontsize=12.5); ax.set_xlim(0, 13200); ax.set_xlabel("使用的擾動數（每個被關掉的基因算一組）")
ax.legend(handles=[Patch(color=v, label=k) for k, v in cmap.items()], frameon=False, loc="lower right", fontsize=12.5, title="細胞類別", title_fontsize=12.5)
fig.tight_layout(); save(fig, "fig_data.png")

# 7. v5 metrics on H1 -----------------------------------------------------------------------------------------
mets = [("區分度", 0.871), ("方向正確", 0.548), ("清單重疊", 0.202), ("整體誤差", 0.165), ("方向排序", 0.075), ("變化大小", -0.022)]
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ys = list(range(len(mets)))[::-1]
ax.barh(ys, [m[1] for m in mets], color=[NAVY if m[1] > 0.4 else (TEAL if m[1] > 0.1 else ORANGE) for m in mets], height=0.6)
for y, m in zip(ys, mets): ax.text(max(m[1], 0) + 0.015, y, f"{m[1]:.3f}", va="center", fontsize=13, color="#2B2D42", fontweight="bold")
ax.axvline(0, color="#2B2D42", lw=1); ax.axvline(1, color=GRAY, lw=1.2, ls="--")
ax.text(1, len(mets) - 0.35, "實驗重複水準 = 1", ha="right", fontsize=12, color=GRAY); ax.text(0.01, -0.85, "基準 = 0", fontsize=12, color=GRAY)
ax.set_yticks(ys); ax.set_yticklabels([m[0] for m in mets]); ax.set_xlim(-0.1, 1.05); ax.set_ylim(-1.1, len(mets) - 0.2)
fig.tight_layout(); save(fig, "fig_metrics_v5.png")

# 8. control-axes sensitivity ---------------------------------------------------------------------------------
axes_n = [0, 5, 10, 20, 30, 50]
pds = {"H1（深度接近期末考）": [0.906, 0.911, 0.909, 0.908, 0.910, 0.913], "HCT116": [0.705, 0.747, 0.755, 0.763, 0.766, 0.768],
       "KOLF200": [0.598, 0.656, 0.657, 0.656, 0.655, 0.654]}
mse = {"H1（深度接近期末考）": [0.832, 0.840, 0.848, 0.858, 0.863, 0.868], "HCT116": [0.997, 1.028, 1.048, 1.063, 1.066, 1.069],
       "KOLF200": [1.109, 1.104, 1.105, 1.106, 1.107, 1.107]}
lc = {"H1（深度接近期末考）": NAVY, "HCT116": TEAL, "KOLF200": "#B5838D"}
fig, axes2 = plt.subplots(1, 2, figsize=(11, 4.2))
for k in pds:
    axes2[0].plot(axes_n, pds[k], marker="o", color=lc[k], lw=2.2, label=k)
    axes2[1].plot(axes_n, mse[k], marker="o", color=lc[k], lw=2.2, label=k)
for a in axes2:
    a.axvline(5, color=ORANGE, ls="--", lw=1.4); a.set_xlabel("移除的對照組方向數"); a.set_xticks(axes_n)
axes2[0].text(5.8, 0.62, "v5 選 5", color=ORANGE, fontsize=13)
axes2[0].set_title("區分度（越高越好）"); axes2[1].set_title("整體誤差（越低越好）")
axes2[0].legend(frameon=False, fontsize=12, loc="lower right")
fig.tight_layout(w_pad=3); save(fig, "fig_ctrl_axes.png")

# 9. GeneNet4: measured vs never-perturbed genes in hidden cell types ------------------------------------------
fig, axes3 = plt.subplots(1, 2, figsize=(8.4, 4.2))
gl = ["有測量的基因", "沒測過的基因"]
b = axes3[0].bar(gl, [0.152, 0.025], color=[NAVY, ORANGE], width=0.5); bar_labels(axes3[0], b, dy=0.003)
axes3[0].set_ylim(0, 0.18); axes3[0].set_yticks([0, 0.05, 0.10, 0.15]); axes3[0].set_title("方向相似度")
b = axes3[1].bar(gl, [0.766, 0.565], color=[NAVY, ORANGE], width=0.5); bar_labels(axes3[1], b, dy=0.004)
axes3[1].set_ylim(0.45, 0.8); axes3[1].axhline(0.5, color=GRAY, ls="--", lw=1.4)
axes3[1].text(0.5, 0.507, "亂猜 = 0.5", color=GRAY, ha="center", fontsize=12); axes3[1].set_title("區分度")
fig.tight_layout(w_pad=3); save(fig, "fig_genenet.png")
print("figures written to", OUT)
