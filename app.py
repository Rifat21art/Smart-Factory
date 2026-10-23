"""Smart Factory AI Tool (Streamlit). Start: double-click run_app.bat (Windows) or ./run_app.sh, or `streamlit run app.py`."""
import os, sys, subprocess, tempfile
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "addon"))
import streamlit as st, pandas as pd
import factory_toolkit as ft
st.set_page_config(page_title="Smart Factory AI Tool", layout="wide")
st.title("Smart Factory AI Tool")
st.caption("Compare dispatching and maintenance policies, test them on your own jobs, and score machine failure risk.")

@st.cache_resource(show_spinner="Training the failure model on AI4I data (first start only)...")
def model(): return ft.failure_model()
model()
from sfs import load_ai4i
src = load_ai4i()[1]
with st.sidebar:
    st.header("Data")
    st.write(f"Failure model trained on: **{src}**")
    if "fallback" in src and st.button("Download real AI4I data (needs internet)"):
        r = subprocess.run([sys.executable, os.path.join(H, "data", "download_data.py")], capture_output=True, text=True)
        st.write(r.stdout or r.stderr); ft._FM = None; st.cache_resource.clear(); st.rerun()
    st.header("About")
    st.write("Simulated job shop; failures follow AI4I 2020 rules. Policies: FIFO, SPT, EDD, CR, ATC, periodic maintenance, AI predictive maintenance (AI-PdM).")

POLS = list(ft.DEFAULT_POLICIES)
tabs = st.tabs(["1 What-if simulator", "2 Your job list", "3 Machine risk scorer", "4 Experiment results"])

with tabs[0]:
    st.subheader("Build a factory and compare policies")
    c = st.columns(5)
    nm = c[0].slider("Machines", 3, 12, 6); load = c[1].slider("Load (0.9 = busy)", 0.6, 1.1, 0.9, 0.05)
    due = c[2].slider("Due-date looseness", 2.0, 8.0, 5.0, 0.5); wr = c[3].slider("Tool wear rate", 0.1, 0.5, 0.2, 0.05); rep = c[4].slider("Mean repair (min)", 10, 120, 40, 5)
    sel = st.multiselect("Policies", POLS, default=["EDD", "ATC", "ATC + periodic PM", "ATC + AI-PdM"], key="p1")
    n = st.slider("Replications (more = slower, more reliable)", 2, 20, 5, key="n1")
    if st.button("Run simulation", key="b1") and sel:
        with st.spinner("Simulating..."):
            runs, s = ft.compare(None, nm, n, load, {k: ft.DEFAULT_POLICIES[k] for k in sel}, due_k=(max(1.0, due - 2), due + 2), wear_rate=wr, repair=float(rep))
        st.dataframe(s, use_container_width=True)
        a, b = st.columns(2); a.write("Mean tardiness (min), lower is better"); a.bar_chart(s["mean_tard"]); b.write("Machine failures per run"); b.bar_chart(s["failures"])
        st.download_button("Download results CSV", s.to_csv().encode(), "whatif_results.csv")

with tabs[1]:
    st.subheader("Test policies on your own jobs")
    st.write("CSV columns: job_id, arrival, due, route (machine ids, e.g. 0-3-2), proc (minutes, e.g. 12-30-18).")
    sample = os.path.join(H, "addon", "sample_jobs.csv")
    if os.path.exists(sample): st.download_button("Download a sample job file", open(sample, "rb").read(), "sample_jobs.csv")
    f = st.file_uploader("Upload job list", type="csv")
    c = st.columns(2); nm2 = c[0].number_input("Number of machines", 2, 30, 6); n2 = c[1].slider("Replications", 2, 20, 5, key="n2")
    sel2 = st.multiselect("Policies", POLS, default=["EDD", "ATC", "ATC + AI-PdM"], key="p2")
    if f and sel2 and st.button("Compare on my jobs", key="b2"):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as t: t.write(f.read()); path = t.name
        try:
            with st.spinner("Simulating..."):
                _, s = ft.compare(ft.read_jobs(path), int(nm2), n2, policies={k: ft.DEFAULT_POLICIES[k] for k in sel2})
            st.dataframe(s, use_container_width=True); st.bar_chart(s["mean_tard"]); st.download_button("Download results CSV", s.to_csv().encode(), "my_jobs_results.csv")
        except Exception as e: st.error(f"Could not run: {e}. Check the column names and that machine ids are below the number of machines.")

with tabs[2]:
    st.subheader("Failure risk for machine readings")
    st.write("CSV columns: type (L/M/H), air, proc, rpm, torque, wear (AI4I column names also work). Flag = risk above 0.05.")
    sr = os.path.join(H, "addon", "sample_readings.csv")
    if os.path.exists(sr): st.download_button("Download sample readings", open(sr, "rb").read(), "sample_readings.csv")
    g = st.file_uploader("Upload readings", type="csv", key="g")
    if g:
        try:
            o = ft.score_readings(pd.read_csv(g)).sort_values("risk", ascending=False)
            st.dataframe(o, use_container_width=True); st.write(f"{int(o.flag.sum())} of {len(o)} readings flagged"); st.bar_chart(o["risk"].reset_index(drop=True))
            st.download_button("Download scores CSV", o.to_csv(index=False).encode(), "risk_scores.csv")
        except Exception as e: st.error(f"Could not score: {e}")

with tabs[3]:
    st.subheader("Results from the experiment")
    R, F = os.path.join(H, "results"), os.path.join(H, "figures")
    if os.path.exists(f"{R}/summary_main.csv"):
        st.write("Main comparison, load 0.9, 20 test seeds"); st.dataframe(pd.read_csv(f"{R}/summary_main.csv", index_col=0), use_container_width=True)
    if os.path.exists(f"{R}/stats_vs_baselines.csv"):
        s = pd.read_csv(f"{R}/stats_vs_baselines.csv"); st.write("Improvement in mean tardiness against ATC (positive = better)")
        st.dataframe(s[(s.metric == "mean_tard") & (s.baseline == "ATC")].drop(columns=["baseline", "metric"]), use_container_width=True)
    caps = {"fig0_architecture": "System architecture", "fig1_main_results": "Main results", "fig2_load_sweep": "Effect of load", "fig3_ablation": "Contribution of each component",
            "fig4_model_quality": "Model quality", "fig5_feature_importance": "Feature importance", "fig6_wip": "Work in process", "fig7_gantt": "Gantt window",
            "fig8_instances": "Other factory instances", "fig9_robustness": "Robustness to model mismatch"}
    for k, v in caps.items():
        if os.path.exists(f"{F}/{k}.png"): st.image(f"{F}/{k}.png", caption=v)
