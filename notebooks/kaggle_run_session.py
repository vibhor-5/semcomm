"""
Kaggle notebook cell — paste everything below into ONE cell.
This script will run an entire session (multiple experiments sequentially).
The only thing you ever change is the ── CONFIG block at the top.
"""

# ════════════════════════════════════════════════════════════════════════════
# ── CHANGE ONLY THIS BLOCK ──────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

GITHUB_REPO = "https://github.com/YOUR_USERNAME/semcomm.git"  # ← your repo URL
SESSION      = 3                                              # ← which session to run (1-18)

# Leave these as-is unless you have a reason to change them
RESUME       = False   # True → continue interrupted experiments from their last checkpoints
DRY_RUN      = False   # True → print what would run without actually training

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

# If private repo: add your GitHub PAT as a Kaggle Secret named GITHUB_TOKEN
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


# ── 4. Run Session ────────────────────────────────────────────────────────
print("=" * 60)
print(f"STEP 4 / 5  Running SESSION {SESSION}")
print("=" * 60)

t0 = time.time()

cmd = [
    sys.executable, "scripts/run_session.py",
    "--session",    str(SESSION),
    "--data_dir",   str(DATA_DIR),
    "--output_dir", str(OUT_DIR),
]
if RESUME:
    cmd.append("--resume")
if DRY_RUN:
    cmd.append("--dry_run")

ret = subprocess.run(cmd, cwd=str(REPO_DIR))
elapsed = time.time() - t0

if ret.returncode != 0:
    print(f"\n✗ Session FAILED (exit code {ret.returncode}).")
    print("  Check the output above for the error message.")
else:
    print(f"\n✓ Session finished in {elapsed/60:.1f} min.\n")


# ── 5. Display results ───────────────────────────────────────────────────────
print("=" * 60)
print("STEP 5 / 5  Results Summary")
print("=" * 60)

# Show summary CSV containing results for all finished experiments
csv_path = OUT_DIR / "summary_table.csv"
if csv_path.exists():
    import csv
    print(f"All extracted metrics metrics ({csv_path.name}):\n")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if rows:
        header = list(rows[0].keys())
        print("  " + "  ".join(f"{h[:12]:<12}" for h in header))
        print("  " + "-" * (14 * len(header)))
        for row in rows:
            print("  " + "  ".join(f"{str(v)[:12]:<12}" for v in row.values()))
else:
    print("(No summary table found. Check if experiments completed successfully.)")

print("\n" + "=" * 60)
print("Outputs are inside:")
print(f"  {REPO_DIR}")
print("Download from:  Notebook → Output tab  (after session ends)")
print("=" * 60)
