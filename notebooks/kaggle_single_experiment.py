"""
Kaggle notebook cell — paste everything below into ONE cell.
The only thing you ever change is the ── CONFIG block at the top.
"""

# ════════════════════════════════════════════════════════════════════════════
# ── CHANGE ONLY THIS BLOCK ──────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

GITHUB_REPO = "https://github.com/YOUR_USERNAME/semcomm.git"  # ← your repo URL
EXPERIMENT   = "configs/flow/B1_linear.yaml"                  # ← which experiment

# Leave these as-is unless you have a reason to change them
RESUME       = False   # True → continue from last saved checkpoint
EVAL_ONLY    = False   # True → skip training, load best.pt and evaluate

# ════════════════════════════════════════════════════════════════════════════
# ── EVERYTHING BELOW IS AUTOMATIC ───────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

import os, sys, subprocess, shutil, json, time
from pathlib import Path

WORK_DIR  = Path("/kaggle/working")
REPO_DIR  = WORK_DIR / "semcomm"
DATA_DIR  = REPO_DIR / "data"
OUT_DIR   = REPO_DIR / "outputs"

# ── 1. Install dependencies ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 / 5  Installing dependencies")
print("=" * 60)

pkgs = [
    "torchdiffeq>=0.2.3",
    "pot>=0.9.0",
    "lpips>=0.1.4",
    "einops>=0.7.0",
    "diffusers>=0.25.0",
    "accelerate>=0.25.0",
    "torchmetrics[image]>=1.0.0",
    "wandb>=0.15.0",
    "git+https://github.com/openai/CLIP.git",
]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + pkgs,
    check=True,
)
print("✓ All packages installed.\n")


# ── 2. Clone / update repo ───────────────────────────────────────────────────
print("=" * 60)
print("STEP 2 / 5  Cloning / updating repo")
print("=" * 60)

# If you have a private repo, add your GitHub PAT as a Kaggle Secret named GITHUB_TOKEN
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    GITHUB_REPO_AUTH = GITHUB_REPO.replace("https://", f"https://{token}@")
except Exception:
    GITHUB_REPO_AUTH = GITHUB_REPO   # public repo — no token needed

if not REPO_DIR.exists():
    subprocess.run(["git", "clone", GITHUB_REPO_AUTH, str(REPO_DIR)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=True)

os.chdir(str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR))
print(f"✓ Repo ready at {REPO_DIR}\n")


# ── 3. Generate configs (idempotent) ─────────────────────────────────────────
print("=" * 60)
print("STEP 3 / 5  Generating experiment configs")
print("=" * 60)

subprocess.run([sys.executable, "scripts/generate_configs.py"], check=True)
print("✓ Configs ready.\n")


# ── 4. Run experiment ────────────────────────────────────────────────────────
print("=" * 60)
print(f"STEP 4 / 5  Running: {EXPERIMENT}")
print("=" * 60)

t0 = time.time()

cmd = [
    sys.executable, "scripts/run_experiment.py",
    "--config",     EXPERIMENT,
    "--data_dir",   str(DATA_DIR),
    "--output_dir", str(OUT_DIR),
]
if RESUME:
    cmd.append("--resume")
if EVAL_ONLY:
    cmd.append("--eval_only")

ret = subprocess.run(cmd, cwd=str(REPO_DIR))
elapsed = time.time() - t0

if ret.returncode != 0:
    print(f"\n✗ Experiment FAILED (exit code {ret.returncode}).")
    print("  Check the output above for the error message.")
else:
    print(f"\n✓ Experiment finished in {elapsed/60:.1f} min.\n")


# ── 5. Display results & checkpoint info ─────────────────────────────────────
print("=" * 60)
print("STEP 5 / 5  Results")
print("=" * 60)

import yaml

with open(EXPERIMENT) as f:
    cfg = yaml.safe_load(f)

exp_id   = cfg["experiment_id"]
exp_dir  = OUT_DIR / exp_id
ckpt_dir = exp_dir / "checkpoints"

# Show metrics
metrics_path = exp_dir / "metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
    print(f"\nMetrics for  {exp_id}:")
    print(f"  {'Metric':<20}  Value")
    print(f"  {'-'*30}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20}  {v:.4f}")
        else:
            print(f"  {k:<20}  {v}")
else:
    print(f"(No metrics.json found at {metrics_path})")

# Show checkpoint files
print(f"\nCheckpoints saved in:\n  {ckpt_dir}")
if ckpt_dir.exists():
    ckpts = sorted(ckpt_dir.glob("*.pt"))
    for ck in ckpts:
        size_mb = ck.stat().st_size / 1e6
        print(f"  {ck.name:<25}  {size_mb:.1f} MB")
else:
    print("  (no checkpoints yet)")

# Show summary CSV
csv_path = OUT_DIR / "summary_table.csv"
if csv_path.exists():
    import csv
    print(f"\nAll results so far ({csv_path.name}):")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if rows:
        header = list(rows[0].keys())
        print("  " + "  ".join(f"{h[:12]:<12}" for h in header))
        print("  " + "-" * (14 * len(header)))
        for row in rows:
            print("  " + "  ".join(f"{str(v)[:12]:<12}" for v in row.values()))

print("\n" + "=" * 60)
print("All outputs are inside:")
print(f"  {REPO_DIR}")
print("Kaggle saves everything in /kaggle/working/ as session output.")
print("Download it from:  Notebook → Output tab  (after the session ends)")
print("=" * 60)
