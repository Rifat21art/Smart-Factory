"""Reusable toolkit on top of src/sfs.py (nothing in the original project is changed).
  python factory_toolkit.py sample                      write sample_jobs.csv and sample_readings.csv
  python factory_toolkit.py jobs my_jobs.csv --machines 6 --repeats 10    compare dispatching/maintenance policies on YOUR job list
  python factory_toolkit.py risk my_readings.csv        failure-risk score for machine readings
jobs CSV columns : job_id, arrival, due, route (machine ids joined by '-', e.g. 0-3-2), proc (minutes joined by '-', e.g. 12-30-18)
readings CSV     : type (L/M/H), air, proc, rpm, torque, wear   (AI4I column names also accepted)"""
import os, sys, copy, argparse, warnings; warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "src"))
import numpy as np, pandas as pd
import sfs
from sfs import CFG, simulate, fvec, load_ai4i, train_failure_model, gen_jobs

_FM = None
def failure_model():
    """Gradient-boosting failure model trained on data/ai4i2020.csv (falls back to the synthetic copy)."""
    global _FM
    if _FM is None: _FM = train_failure_model(load_ai4i()[0])[0]
    return _FM

DEFAULT_POLICIES = {
    "FIFO": dict(rule="FIFO"), "SPT": dict(rule="SPT"), "EDD": dict(rule="EDD"), "CR": dict(rule="CR"),
    "ATC": dict(rule="ATC", kk=2.0),
    "ATC + periodic PM": dict(rule="ATC", kk=2.0, pdm="periodic", per_wear=50),
    "EDD + AI-PdM": dict(rule="EDD", pdm="predictive", thr=0.05),
    "ATC + AI-PdM": dict(rule="ATC", kk=2.0, pdm="predictive", thr=0.05)}

def read_jobs(path):
    d = pd.read_csv(path); jobs = []
    for _, r in d.iterrows():
        route = np.array([int(x) for x in str(r["route"]).split("-")]); p = np.array([float(x) for x in str(r["proc"]).split("-")])
        assert len(route) == len(p), f"job {r['job_id']}: route and proc lengths differ"
        jobs.append(dict(arr=float(r["arrival"]), route=route, p=p, due=float(r["due"]), op=0, done=None))
    return sorted(jobs, key=lambda j: j["arr"])

def compare(jobs=None, n_mach=6, repeats=10, load=0.9, policies=None, warmup_frac=0.15, seeds=None, **cfg_over):
    """Run policies with common random numbers. jobs=None -> generated jobs (use load, due_k ... in cfg_over); else your job list.
    Returns (per-run DataFrame, summary DataFrame with mean and 95% CI)."""
    policies = policies or DEFAULT_POLICIES; cfg = {**CFG, "n_mach": n_mach, **cfg_over}; fm = failure_model()
    if jobs is not None:
        cfg["horizon"] = max(j["arr"] for j in jobs) + 1.0; cfg["warmup"] = warmup_frac * cfg["horizon"]
        sfs.gen_jobs = lambda c, s, l: copy.deepcopy(jobs)
    else: sfs.gen_jobs = gen_jobs
    rows = []
    try:
        for s in (seeds or range(1, repeats + 1)):
            for name, pol in policies.items():
                rows.append(dict(policy=name, seed=s, **simulate(cfg, s, load, pol, fail_model=fm)["res"]))
    finally: sfs.gen_jobs = gen_jobs
    runs = pd.DataFrame(rows); cols = ["mean_tard", "pct_tardy", "mean_flow", "throughput", "failures", "downtime"]
    g = runs.groupby("policy", sort=False); summ = g[cols].mean().round(2)
    summ["mean_tard_ci95"] = (1.96 * g["mean_tard"].std() / np.sqrt(g["mean_tard"].count())).round(2)
    return runs, summ

def score_readings(df):
    """Failure probability for rows of machine readings. Returns df with columns risk and flag (risk > 0.05)."""
    ren = {"Type": "type", "Air temperature [K]": "air", "Process temperature [K]": "proc", "Rotational speed [rpm]": "rpm",
           "Torque [Nm]": "torque", "Tool wear [min]": "wear"}
    d = df.rename(columns=ren).copy(); fm = failure_model()
    X = np.array([fvec(str(r["type"]), r["air"], r["proc"], r["rpm"], r["torque"], r["wear"]) for _, r in d.iterrows()])
    d["risk"] = fm.predict_proba(X)[:, 1].round(4); d["flag"] = d["risk"] > 0.05; return d

def write_samples(n_jobs=150):
    j = gen_jobs({**CFG, "horizon": 3000}, 7, 0.9)[:n_jobs]
    pd.DataFrame([dict(job_id=i, arrival=round(x["arr"], 1), due=round(x["due"], 1), route="-".join(map(str, x["route"])),
                       proc="-".join(f"{v:.1f}" for v in x["p"])) for i, x in enumerate(j)]).to_csv("sample_jobs.csv", index=False)
    rng = np.random.default_rng(3); rows = []
    for _ in range(40):
        a, p, r, t = sfs.sample_cond(rng); rows.append(dict(type=rng.choice(list("LMH")), air=a, proc=p, rpm=r, torque=t, wear=rng.uniform(0, 240)))
    pd.DataFrame(rows).round(2).to_csv("sample_readings.csv", index=False); print("wrote sample_jobs.csv, sample_readings.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sample", "jobs", "risk"]); ap.add_argument("file", nargs="?")
    ap.add_argument("--machines", type=int, default=6); ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "sample": write_samples()
    elif a.cmd == "jobs":
        runs, summ = compare(read_jobs(a.file), a.machines, a.repeats); print(summ.to_string()); runs.to_csv("jobs_runs.csv", index=False); summ.to_csv("jobs_summary.csv")
    else:
        out = score_readings(pd.read_csv(a.file)); print(out.to_string()); out.to_csv("risk_scores.csv", index=False)
