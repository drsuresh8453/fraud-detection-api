"""Generates sagemaker/retrain_pipeline.ipynb (the notebook SageMaker runs when drift is detected).
Usage: python -m scripts.build_retrain_notebook"""
import nbformat as nbf

BY = "**Created by: Suresh D R | AI Product Developer & Technology Mentor**"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s, tags=None: cells.append(nbf.v4.new_code_cell(s.strip(), metadata={"tags": tags or []}))

md(f"""
# 🔁 Automatic Retraining — Fraud Detection Model
{BY}

This notebook is **started automatically** by `monitoring/trigger_retrain.py` when the weekly drift check
(Evidently AI) finds a problem. SageMaker runs it with **papermill** inside our Docker image:

1. pulls the **latest code from GitHub** (so retraining always uses the newest `preprocess.py` / `features.py` / `train.py`)
2. reads the **history + new labelled data** from S3
3. trains a new model (challenger) and compares it with the current model (champion) on the newest mature month
4. **promotes** the challenger only if it is better → the API pods load it automatically
5. this notebook, with all outputs, is saved to `s3://<bucket>/reports/retraining/<job>/` as evidence for auditors

You can also open it in **SageMaker Studio** and run it by hand.
""")
code('''
# Parameters (papermill overrides these)
drift_report = ""      # S3 key of the drift summary that triggered this run
promote = True         # save the new model if it wins
''', tags=["parameters"])
md("## Step 1 — Setup")
code('''
import os, sys, json, subprocess
sys.path.insert(0, os.path.abspath(".")); sys.path.insert(0, os.path.abspath(".."))
import pandas as pd
from src import config, storage
from src.train import retrain, notify_github
pd.set_option("display.width", 160)
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    commit = "code baked into the image"
print(f"Storage: {config.STORAGE_MODE} | bucket: {config.S3_BUCKET} | code version: {commit}")
''')
md("## Step 2 — Why are we retraining? (drift report)")
code('''
if drift_report and storage.exists(drift_report):
    s = storage.read_json(drift_report)
    print("Model:", s["model_version"], "| date:", s["date"])
    print("Reasons:"); [print("  •", r) for r in s["reasons"]]
    print("Drifted features (PSI):", s["drifted_features"])
    print("Performance:", s["performance"])
else:
    print("Manual run (no drift report given).")
''')
md("""
## Step 3 — Retrain: champion vs challenger
`src/train.py` repeats Notebooks 2 → 3 → 6 → 7 automatically:
cleaning + data-quality gate → time split on mature labels → fill values & encoders on train only → LightGBM →
threshold by business cost (≤ 3 % alerts) → comparison on the newest mature month.
""")
code('''
report = retrain(promote=promote, reason=", ".join(s["reasons"]) if drift_report and storage.exists(drift_report) else "manual")
''')
md("## Step 4 — Result")
code('''
comp = pd.DataFrame({"champion (current)": report["champion"], "challenger (new)": report["challenger"]})
display(comp)
print("Decision:", report["decision"].upper(), "| new version:", report["new_version"])
''')
code('''
if report["decision"] == "promote":
    cfg = storage.read_json(config.MODEL_CONFIG_KEY)
    print(f"Live model is now {cfg['model_version']} | threshold {cfg['threshold']} | bands {cfg['bands']}")
    print(notify_github(cfg["model_version"]))     # optional: GitHub Actions restarts the pods right away
else:
    print("Current model kept — the challenger was not better. Investigate the drift report before the next run.")
''')
md("""
## ✅ Done
- **Promoted:** API pods pick up the new model within `MODEL_RELOAD_SECONDS` (or immediately via the GitHub workflow).
- **Not promoted:** the current model stays; the drift alert is still open for the data-science team.
- The next drift check compares against the **new reference** saved by `train.py`.
""")
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, "sagemaker/retrain_pipeline.ipynb")
print("written sagemaker/retrain_pipeline.ipynb")
