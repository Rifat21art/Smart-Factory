# Smart Factory AI Tool

Simulation tool for comparing dispatching and maintenance policies in a job shop whose machine failures follow the AI4I 2020 failure modes.

## Background and findings (load 0.9, 20 test seeds)
Job-shop delivery performance depends on which job a free machine takes next and on when machines are maintained. This tool evaluates both decisions together: five classical dispatching rules, fixed-schedule maintenance, a learned failure-risk model that triggers maintenance (AI-PdM), and a learned tardiness model that adjusts ATC dispatching.

- AI-PdM with ATC reduces mean tardiness by about 82% against tuned ATC and cuts failures from 176 to 62 per run.
- Fixed-schedule maintenance is worse than no maintenance.
- Learned dispatching gives no gain over tuned ATC.
- A rule-based controller using the known failure rules matches the learned failure model.
- The gain holds in two further factory instances and under shorter tool life and higher torque, and is unreliable at 30% sensor noise.

## Validation
Simulation-based. All methods are tuned on separate validation seeds and compared on held-out test seeds with common random numbers, paired Wilcoxon tests and bootstrap intervals. Models are scored on held-out data (failure model AUC about 0.99). Robustness is tested by changing the simulated machines while the model stays fixed. There is no validation on real plant data.

## Use
Requires Python 3.9+. Windows: double-click `run_app.bat`. Mac/Linux: `./run_app.sh`. The first start installs the packages and opens the tool in the browser.

| Tab | Function |
|---|---|
| What-if simulator | Set machines, load, due dates, wear and repair times; compare policies |
| Your job list | Upload a CSV (`job_id, arrival, due, route, proc`) and compare policies |
| Machine risk scorer | Upload readings (`type, air, proc, rpm, torque, wear`) to get failure risk |
| Experiment results | Tables and figures of the full experiment |

## Reproduce and update
`pip install jupyter`, then run `Smart_Factory_Experiment.ipynb` (Restart & Run All, about 35 min; `QUICK = True` for a test). Outputs are written to `results/` and `figures/`. `python addon/extra_baselines.py` regenerates the extra baselines. To change a setting, edit `CFG` in `src/sfs.py` or the scenario lists in the notebook, then re-run.

## Limitations
The failure model is trained on AI4I data, so risk scores apply only to similar machines. The simulator shares its failure rules with the data, so gains are probably optimistic. Jobs are generated, not taken from a plant.

## Data and citation
S. Matzka, "Explainable artificial intelligence for predictive maintenance applications," AI4I 2020; UCI Machine Learning Repository, dataset 601 (CC BY 4.0). Cite this repository as: [author, year, URL].
