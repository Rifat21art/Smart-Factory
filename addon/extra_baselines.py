"""Extra baselines a reviewer will ask for: does the ML failure model beat PdM that uses the known failure physics, or a wear limit alone?
Writes results/extra_baselines.csv and results/extra_baselines_summary.csv. Seeds 1000+ match the main experiment (paired)."""
import os, sys; H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "src"))
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sfs import CFG, OSF_LIM, simulate
class RulePdM:      # PdM from the documented AI4I failure rules (needs engineering knowledge, no learning)
    def predict_proba(self, X):
        t, air, proc, rpm, tq, wear, dT, pw, wt = X[0]; lim = list(OSF_LIM.values())[int(t)]
        p = float((dT < 8.6 and rpm < 1380) or pw < 3500 or pw > 9000 or wt > lim or wear >= 200); return np.array([[1 - p, p]])
class WearPdM:      # tool change when wear passes a limit, ignores operating conditions
    def __init__(s, w): s.w = w
    def predict_proba(self, X): p = float(X[0][5] >= self.w); return np.array([[1 - p, p]])
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
POL = {"ATC + rule-based PdM": ("ATC", RulePdM()), "ATC + wear-limit PdM (150)": ("ATC", WearPdM(150)), "ATC + wear-limit PdM (200)": ("ATC", WearPdM(200))}
rows = [dict(policy=n, seed=s, **simulate(CFG, s, 0.9, dict(rule=r, kk=2.0, pdm="predictive", thr=0.05), fail_model=m)["res"])
        for s in range(1000, 1000 + N) for n, (r, m) in POL.items()]
R = os.path.join(H, "..", "results"); os.makedirs(R, exist_ok=True); d = pd.DataFrame(rows); d.to_csv(f"{R}/extra_baselines.csv", index=False)
print(d.groupby("policy")[["mean_tard", "pct_tardy", "failures", "downtime"]].mean().round(2))
