"""Smart Factory Simulator: stochastic job shop + AI production control.
Pure numpy/sklearn (no SimPy needed). Time unit = minutes."""
import heapq, math, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TYPES = ["L", "L", "M", "M", "H", "M"]           # machine quality class (AI4I variants)
OSF_LIM = {"L": 11000, "M": 12000, "H": 13000}

# ---------- machine physics: AI4I 2020 documented failure modes ----------
def sample_cond(rng):
    air = rng.normal(300, 2)
    proc = air + 10 + rng.normal(0, 1)
    rpm = 1168 + rng.gamma(4, 95)
    torque = max(3.0, rng.normal(40, 6.5) - 0.01 * (rpm - 1540))
    return air, proc, rpm, torque

def truth_fail(typ, air, proc, rpm, torque, wear, tw_thr, rng):
    power = torque * rpm * 2 * math.pi / 60
    if wear >= tw_thr: return "TWF"
    if proc - air < 8.6 and rpm < 1380: return "HDF"
    if power < 3500 or power > 9000: return "PWF"
    if wear * torque > OSF_LIM[typ]: return "OSF"
    if rng.random() < 0.001: return "RNF"
    return None

def fvec(typ, air, proc, rpm, torque, wear):
    pw = torque * rpm * 2 * math.pi / 60
    return [{"L": 0, "M": 1, "H": 2}[typ], air, proc, rpm, torque, wear, proc - air, pw, wear * torque]

def make_ai4i_like(n=10000, seed=0):
    """Fallback synthetic copy of AI4I generative rules (only used if the real CSV is absent)."""
    rng = np.random.default_rng(seed); rows = []
    for _ in range(n):
        typ = rng.choice(["L", "M", "H"], p=[.6, .3, .1])
        a, p, r, t = sample_cond(rng); w = 253 * rng.beta(1.6, 2.1)
        f = truth_fail(typ, a, p, r, t, w, rng.uniform(200, 240), rng)
        rows.append([typ, a, p, r, t, w, int(f is not None)])
    return pd.DataFrame(rows, columns=["Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"])

def load_ai4i():
    real = os.path.join(DATA, "ai4i2020.csv")
    if os.path.exists(real): return pd.read_csv(real), "UCI AI4I 2020 (real)"
    syn = os.path.join(DATA, "ai4i_like_synthetic.csv")
    if not os.path.exists(syn): make_ai4i_like().to_csv(syn, index=False)
    return pd.read_csv(syn), "AI4I-like synthetic (fallback)"

def train_failure_model(df, seed=0):
    X = np.array([fvec(r["Type"], r["Air temperature [K]"], r["Process temperature [K]"],
                       r["Rotational speed [rpm]"], r["Torque [Nm]"], r["Tool wear [min]"])
                  for _, r in df.iterrows()])
    y = df["Machine failure"].values
    rs = np.random.default_rng(seed).permutation(len(y)); cut = int(.75 * len(y))
    tr, te = rs[:cut], rs[cut:]
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed).fit(X[tr], y[tr])
    return m, (X[te], y[te])

# ---------- jobs ----------
def gen_jobs(cfg, seed, load):
    rng = np.random.default_rng(seed)
    lam = load * cfg["n_mach"] * cfg["eff"] / (4.0 * cfg["mean_p"])
    t, jobs = 0.0, []
    while t < cfg["horizon"]:
        t += rng.exponential(1 / lam)
        k = int(rng.integers(3, 6))
        route = rng.permutation(cfg["n_mach"])[:k]
        p = np.clip(rng.lognormal(math.log(cfg["mean_p"]) - .125, .5, k), 5, 80)
        due = t + rng.uniform(*cfg["due_k"]) * p.sum()
        jobs.append(dict(arr=t, route=route, p=p, due=due, op=0, done=None))
    return jobs

# ---------- dispatching ----------
CLASSIC = ["FIFO", "SPT", "EDD", "CR", "ATC"]
def classic_key(rule, j, t, qin, pbar, kk):
    p = j["p"][j["op"]]; rem = j["p"][j["op"]:].sum()
    if rule == "FIFO": return qin
    if rule == "SPT": return p
    if rule == "EDD": return j["due"]
    if rule == "CR": return (j["due"] - t) / rem
    slack = max(j["due"] - t - rem, 0)                       # ATC
    return -(1 / p) * math.exp(-slack / (kk * pbar))

JF = ["slack", "remaining_work", "remaining_ops", "op_time", "waited", "queue_len", "next_queue_len", "wip", "time_to_due"]
def job_feats(j, t, qin, qlen, nextq, wip):
    p = j["p"][j["op"]]; rem = j["p"][j["op"]:].sum()
    return [j["due"] - t - rem, rem, len(j["p"]) - j["op"], p, t - qin, qlen, nextq, wip, j["due"] - t]

# ---------- simulation ----------
def simulate(cfg, seed, load, policy, fail_model=None, tard_model=None, collect=False, trace=False):
    """policy = dict(rule, pdm=None|'periodic'|'predictive', thr, eps, kk, per_wear)"""
    nm = cfg["n_mach"]; rule = policy["rule"]; pdm = policy.get("pdm")
    jobs = gen_jobs(cfg, seed, load)
    mrng = [np.random.default_rng([seed, 7, m]) for m in range(nm)]
    wear = np.zeros(nm); tw = [mr.uniform(200, 240) * cfg.get("life", 1.0) for mr in mrng]
    queue = [[] for _ in range(nm)]; busy = [False] * nm
    ev = [(j["arr"], 0, i) for i, j in enumerate(jobs)]; heapq.heapify(ev)
    wip = 0; busy_time = np.zeros(nm); st = dict(fail=0, pm=0, down=0.0)
    wip_ts, gantt, rows = [], [], []; pbar = cfg["mean_p"]

    def dispatch(m, t):
        if busy[m] or not queue[m]: return
        q = queue[m]
        X = None
        if collect or rule not in CLASSIC:
            X = np.array([job_feats(jobs[i], t, qi, len(q),
                  len(queue[jobs[i]["route"][jobs[i]["op"] + 1]]) if jobs[i]["op"] + 1 < len(jobs[i]["route"]) else 0, wip)
                  for i, qi in q])
            if collect and len(q) > 1:
                for (i, _), x in zip(q, X):
                    if mrng[m].random() < .3: rows.append((i, *x))
        if rule in CLASSIC:
            pick = int(np.argmin([classic_key(rule, jobs[i], t, qi, pbar, policy.get("kk", 2.0)) for i, qi in q]))
        else:
            # AI-dispatch = ATC index whose slack is reduced by the ML-predicted tardiness (a=0 reproduces ATC)
            pt = np.maximum(np.expm1(tard_model.predict(X)), 0)
            keff = np.maximum(X[:, 0] - policy.get("a", 1.0) * pt, 0)
            pick = int(np.argmax(np.exp(-keff / (policy.get("kk", 2.0) * pbar)) / X[:, 3]))
        i, _ = q.pop(pick); j = jobs[i]; p = j["p"][j["op"]]
        typ = TYPES[m % len(TYPES)]; r = mrng[m]
        air, proc, rpm, tq = sample_cond(r); extra = 0.0; action = None
        rpm += cfg.get("rpm_shift", 0.0); tq = max(3.0, tq + cfg.get("torque_shift", 0.0))   # robustness: world differs from training data
        if pdm == "periodic" and wear[m] >= policy.get("per_wear", 40):
            extra, wear[m], tw[m], action = cfg["tool_change"], 0.0, r.uniform(200, 240) * cfg.get("life", 1.0), "PM"
        elif pdm == "predictive":
            nz = cfg.get("sensor_noise", 0.0)                    # robustness: model sees noisy readings
            ob = (rpm * (1 + nz * r.normal()), tq * (1 + nz * r.normal()), max(0.0, wear[m] * (1 + nz * r.normal()))) if nz else (rpm, tq, wear[m])
            ph = fail_model.predict_proba([fvec(typ, air, proc, ob[0], ob[1], ob[2])])[0, 1]
            if ph > policy.get("thr", 0.2):
                if wear[m] > 120:                      # wear-driven risk -> tool change
                    extra, wear[m], tw[m], action = cfg["tool_change"], 0.0, r.uniform(200, 240) * cfg.get("life", 1.0), "PM"
                else:                                                  # condition-driven risk -> set-point correction
                    extra, action = cfg["adjust"], "ADJ"; rpm, tq = 1540.0, 40.0
        if action: st["pm"] += 1
        f = truth_fail(typ, air, proc, rpm, tq, wear[m], tw[m], r)
        dur = p * (1 + 0.1 * (tq - 40) / 40); rep = 0.0
        if f:
            rep = r.lognormal(math.log(cfg["repair"]) - .125, .5); st["fail"] += 1
            if f in ("TWF", "OSF"): wear[m], tw[m] = 0.0, r.uniform(200, 240) * cfg.get("life", 1.0)
        wear[m] += dur * cfg["wear_rate"]            # wear minutes per processing minute
        tot = extra + dur + rep; st["down"] += extra + rep; busy[m] = True; busy_time[m] += tot
        if trace: gantt.append((m, t, extra, dur, rep, i, action, f))
        heapq.heappush(ev, (t + tot, 1, (m, i)))

    while ev:
        t, kind, x = heapq.heappop(ev)
        if t > cfg["horizon"] * 1.6: break
        if kind == 0:
            wip += 1; m = jobs[x]["route"][0]; queue[m].append((x, t)); dispatch(m, t)
        else:
            m, i = x; busy[m] = False; j = jobs[i]; j["op"] += 1
            if j["op"] >= len(j["route"]): j["done"] = t; wip -= 1
            else:
                m2 = j["route"][j["op"]]; queue[m2].append((i, t)); dispatch(m2, t)
            dispatch(m, t)
        if trace: wip_ts.append((t, wip))
    ok = [j for j in jobs if j["arr"] >= cfg["warmup"] and j["done"] is not None]
    tard = np.array([max(0, j["done"] - j["due"]) for j in ok]); flow = np.array([j["done"] - j["arr"] for j in ok])
    res = dict(jobs=len(ok), mean_tard=tard.mean(), pct_tardy=100 * (tard > 0).mean(), p95_tard=np.percentile(tard, 95),
               mean_flow=flow.mean(), throughput=len(ok) / (cfg["horizon"] - cfg["warmup"]) * 60,
               util=100 * busy_time.sum() / (nm * cfg["horizon"]), failures=st["fail"], interventions=st["pm"], downtime=st["down"])
    out = dict(res=res)
    if collect:
        out["train"] = [(*r, max(0, jobs[r[0]]["done"] - jobs[r[0]]["due"]) if jobs[r[0]]["done"] and jobs[r[0]]["arr"] >= cfg["warmup"] else np.nan) for r in rows]
    if trace: out["gantt"] = gantt; out["wip"] = wip_ts
    return out

CFG = dict(n_mach=6, horizon=14000, warmup=2000, mean_p=20, eff=0.85, due_k=(3.0, 7.0),
           wear_rate=0.2, tool_change=15.0, adjust=3.0, repair=40.0)
