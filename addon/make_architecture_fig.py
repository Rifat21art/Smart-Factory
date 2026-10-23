"""Architecture diagram -> figures/fig0_architecture.png (same colours as the other figures)."""
import os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
H = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f"{H}/../figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 4.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 52)
def box(x, y, w, h, t, c, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=c, ec="none")); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8, color=tc)
def arr(x1, y1, x2, y2, t=""):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333")); 
    if t: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.6, t, ha="center", fontsize=6.5, color="#333333")
box(1, 36, 21, 12, "AI4I 2020 data\n(10,000 machine records)", "#999999")
box(27, 36, 22, 12, "Failure-risk model\n(gradient boosting)", "#009E73")
box(1, 18, 21, 12, "Job generator\narrivals, routes,\ntimes, due dates", "#0072B2")
box(27, 12, 46, 24, "", "#EAEAEA", "k"); ax.text(50, 33, "Discrete-event job-shop simulator", ha="center", fontsize=8.5, weight="bold")
box(30, 15, 12, 13, "Queues\nper machine", "#8FA7BF"); box(44, 15, 26, 13, "Machines: wear, torque, speed,\ntemperatures, breakdowns,\nrepairs", "#56B4E9")
box(78, 36, 21, 12, "Tardiness-risk model\n(trained on sim. logs)", "#009E73")
box(78, 15, 21, 18, "Controller\n1 dispatch rule\n(FIFO/SPT/EDD/CR/ATC)\n2 maintenance action\n(tool change, set-point)", "#332288")
box(12, 1, 76, 7, "Metrics: tardiness, tardy jobs, flow time, failures, downtime | paired statistics", "#D55E00")
arr(22, 42, 27, 42, "train"); ax.plot([49, 75.5, 75.5], [43, 43, 30], color="#333333", lw=1.2); arr(75.5, 30, 78, 30, ""); ax.text(62, 44.4, "failure risk", ha="center", fontsize=6.5); arr(88.5, 36, 88.5, 33.4, ""); arr(22, 24, 27, 24, "jobs")
arr(78, 21, 73, 21, ""); ax.text(75.4, 17.8, "actions", ha="center", fontsize=6.5); arr(50, 12, 50, 8.4, ""); ax.text(53, 9.6, "logs", fontsize=6.5)

plt.savefig(f"{H}/../figures/fig0_architecture.png", dpi=300, bbox_inches="tight")
