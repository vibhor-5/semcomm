# Running Semcomm on Kaggle — Complete Guide

> Everything in this repo is designed so that **you clone the repo once and every output
> file (checkpoints, metrics, plots) saves inside the repo directory automatically**.
> No path configuration needed.

---

## Where files go on Kaggle

```
/kaggle/working/
└── semcomm/                     ← cloned repo (THIS is your working dir)
    ├── data/                    ← datasets downloaded here automatically
    │   ├── cifar-10-batches-py/
    │   ├── cifar10_clip_train.pt   ← CLIP token cache (pre-computed once)
    │   └── tinyimagenet/           ← attach as Kaggle Dataset (see below)
    ├── outputs/                 ← ALL results land here
    │   ├── summary_table.csv       ← one row per experiment (auto-appended)
    │   └── B1_linear/
    │       ├── checkpoints/
    │       │   ├── epoch_5.pt
    │       │   ├── epoch_10.pt
    │       │   └── best.pt
    │       ├── metrics.json
    │       └── figures/
    │           ├── rate_curve.png
    │           └── rate_curve.pdf
    └── configs/  models/  ...
```

**Kaggle automatically saves everything in `/kaggle/working/` as output.**  
After a session ends you can download the whole `semcomm/` folder or push it as a
private Kaggle Dataset to reuse in future sessions.

---

## One-time setup (run in the FIRST cell of every new notebook)

```python
# ── 1. Install dependencies ──────────────────────────────────────────────────
import subprocess, sys

deps = [
    "torchdiffeq>=0.2.3",
    "pot>=0.9.0",
    "lpips>=0.1.4",
    "einops>=0.7.0",
    "wandb>=0.15.0",
    "diffusers>=0.25.0",
    "accelerate>=0.25.0",
    "torchmetrics[image]>=1.0.0",
    "git+https://github.com/openai/CLIP.git",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + deps, check=True)
print("Dependencies installed.")

# ── 2. Clone / update the repo ───────────────────────────────────────────────
import os

REPO_URL = "https://github.com/YOUR_USERNAME/semcomm.git"   # ← change this
WORK_DIR = "/kaggle/working"

if not os.path.isdir(f"{WORK_DIR}/semcomm"):
    subprocess.run(["git", "clone", REPO_URL, f"{WORK_DIR}/semcomm"], check=True)
else:
    subprocess.run(["git", "-C", f"{WORK_DIR}/semcomm", "pull"], check=True)

os.chdir(f"{WORK_DIR}/semcomm")
sys.path.insert(0, f"{WORK_DIR}/semcomm")
print("Repo ready. CWD:", os.getcwd())
```

> **Tip — secret token for private repo:**  
> Add your GitHub PAT as a Kaggle Secret named `GITHUB_TOKEN`, then use:
> ```python
> from kaggle_secrets import UserSecretsClient
> token = UserSecretsClient().get_secret("GITHUB_TOKEN")
> REPO_URL = f"https://vibhorkumar:{token}@github.com/YOUR_USERNAME/semcomm.git"
> ```

---

## Running experiments

### Option A — Run a whole session (recommended)

Maps directly to the 18-session plan in the Experiment Catalogue:

```python
# Session 3 = B1 flow path comparison (≈3 hr)
!python scripts/run_session.py --session 3
```

All checkpoints, metrics, and plots land in `./outputs/` automatically.

### Option B — Run a single experiment

```python
!python scripts/run_experiment.py --config configs/flow/B1_linear.yaml
```

### Option C — Resume after a disconnect

```python
# Exactly the same command + --resume
!python scripts/run_session.py --session 3 --resume
# or
!python scripts/run_experiment.py --config configs/flow/B1_linear.yaml --resume
```

The script finds the latest `epoch_N.pt` in `outputs/B1_linear/checkpoints/`
and continues from there — **no manual epoch counting needed**.

### Option D — Eval only (no training)

```python
!python scripts/run_experiment.py --config configs/flow/B4_k128.yaml --eval_only
```

---

## Persisting outputs between Kaggle sessions

Kaggle deletes `/kaggle/working/` when a notebook session ends, BUT it saves
everything as **notebook output** first. Here's how to reuse it:

### Method 1 — Save outputs as a Kaggle Dataset (best)

```python
# At the end of your notebook, after training completes:
import subprocess

# Create/update a private Kaggle Dataset with the outputs folder
subprocess.run([
    "kaggle", "datasets", "version",
    "-p", "/kaggle/working/semcomm/outputs",
    "-m", f"Session 3: B1 results",
    "--dir-mode", "zip",
], check=True)
```

Then in the NEXT session, attach this dataset and symlink it:

```python
# At the top of the next session's notebook:
import os, shutil
DATASET_PATH = "/kaggle/input/semcomm-outputs"   # mounted read-only
WORKING_OUTPUTS = "/kaggle/working/semcomm/outputs"
os.makedirs(WORKING_OUTPUTS, exist_ok=True)

# Copy previous outputs so they're writable and can be extended
if os.path.isdir(DATASET_PATH):
    shutil.copytree(DATASET_PATH, WORKING_OUTPUTS, dirs_exist_ok=True)
    print(f"Loaded {len(list(os.walk(WORKING_OUTPUTS)))} output folders.")
```

### Method 2 — Push to GitHub after each session

```python
# Push outputs (metrics.json, summary_table.csv) but NOT large checkpoints
import subprocess, os

os.chdir("/kaggle/working/semcomm")

# Stage only small files
subprocess.run(["git", "add", "outputs/*/metrics.json",
                "outputs/summary_table.csv"], check=True)
subprocess.run(["git", "commit", "-m", "Session 3 results: B1"], check=True)
subprocess.run(["git", "push"], check=True)
# Large .pt checkpoints stay in Kaggle Dataset (use Method 1 for those)
```

---

## Viewing results

### During training — check loss in real time

```python
# In a new cell while training is running:
import json, os, glob

output_dir = "/kaggle/working/semcomm/outputs"
for metrics_file in sorted(glob.glob(f"{output_dir}/*/metrics.json")):
    exp_id = metrics_file.split("/")[-2]
    with open(metrics_file) as f:
        m = json.load(f)
    print(f"{exp_id:30s}  CLIP={m.get('clip',0):.3f}  "
          f"PSNR={m.get('psnr',0):.1f}  LPIPS={m.get('lpips',0):.3f}")
```

### After all sessions — load the summary table

```python
import pandas as pd

df = pd.read_csv("/kaggle/working/semcomm/outputs/summary_table.csv")
print(df.sort_values("clip", ascending=False).to_string(index=False))
```

### Generate comparison plots

```python
from evaluation.visualise import plot_rate_semantic_curve
import json, glob

# Collect all B4 (flow rate sweep) results
flow_results = []
for f in sorted(glob.glob("outputs/B4_k*/metrics.json")):
    with open(f) as fp:
        m = json.load(fp)
    flow_results.append({'model_name': m['experiment_id'],
                         'bpp': m['bpp'], 'clip': m['clip']})

plot_rate_semantic_curve(
    results          = flow_results,
    baseline_results = [],   # add A2/A3 results here
    save_path        = "outputs/D1_rate_curves/figures/flow_vs_jpeg",
)
```

---

## Session schedule reference

| Session | Experiments | Est. GPU time |
|---------|-------------|---------------|
| 1 | A1, A3 (CPU — baselines) | 30 min |
| 2 | A2 (DeepJSCC k=64, k=128) | 1 hr |
| 3 | B1 (flow path: linear + OT) | 3 hr |
| 4 | B2 (depth ablation: 3/5/8 blocks) | 3 hr |
| 5 | B3 (conditioning ablation) | 6 hr |
| 6 | B4 (rate sweep k=32/64/128/256) | 8 hr |
| 7 | B5, B6 (SNR sweep + channel-aware) | 4.5 hr |
| 8 | B7, B9, B10 (guidance, ODE steps, fading) | 2 hr |
| 9 | B8 (TinyImageNet 64×64, flow) | 6 hr |
| 10 | C1 (diffusion conditioning) | 6 hr |
| 11 | C3 (diffusion rate sweep) | 8 hr |
| 12 | C2, C4, C5, C10 (eval-only diffusion) | 3 hr |
| 13 | C6, C7 (channel-aware + latent space) | 8 hr |
| 14 | C8 (semantic loss augmentation) | 6 hr |
| 15 | C9 (TinyImageNet 64×64, diffusion) | 8 hr |
| 16 | D1–D5 (head-to-head plots) | 3 hr |
| 17 | E1, E2 (encoder + token ablations) | 10 hr |
| 18 | E3, E4 (quant + augmentation ablations) | 14 hr |

**Total: ~115 GPU hours ≈ 4 Kaggle weeks at 30 hr/week.**

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: clip` | Run the install cell; CLIP needs re-installing each session |
| "CUDA out of memory" | Edit the config YAML: reduce `batch_size` to 32 or `n_channels` to 64 |
| Session disconnected mid-training | Re-run the same command with `--resume` |
| Can't find checkpoint | Check `outputs/<experiment_id>/checkpoints/` — list files with `ls` |
| W&B not logging | Add `WANDB_API_KEY` as a Kaggle Secret; the trainer reads it automatically |
| TinyImageNet not found | Attach the TinyImageNet Kaggle Dataset to your notebook; it mounts at `/kaggle/input/tiny-imagenet-200/` |
