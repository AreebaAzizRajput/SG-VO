#!/usr/bin/env python3
"""Install Miniconda + orbsfm env on remote GPU server, handle NVIDIA driver."""
import paramiko, warnings, sys
warnings.filterwarnings("ignore")

HOST = "119.63.132.182"
USER = "areeba"
PASS = "51gm@123++"
REMOTE_DIR = "/home/areeba/SG-VO"
CONDA = "/home/areeba/miniconda3/bin/conda"

def run(ssh, cmd, timeout=300, show=True):
    if show: print(f"\n$ {cmd[:120]}")
    _, stdout, stderr = ssh.exec_command(f"bash -l -c {repr(cmd)}", timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    combined = (out + err).strip()
    if combined and show:
        for line in combined.split('\n')[:20]:
            print("  " + line)
    return combined

def run_raw(ssh, cmd, timeout=300, show=True):
    """Run without bash -l wrapper."""
    if show: print(f"\n$ {cmd[:120]}")
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    combined = (out + err).strip()
    if combined and show:
        for line in combined.split('\n')[:20]:
            print("  " + line)
    return combined

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS, timeout=15)
print("✅ Connected to", HOST)

# ── 1. GPU Diagnosis ──────────────────────────────────────────────────────────
print("\n" + "="*55)
print("1. GPU Diagnosis")
print("="*55)
run_raw(ssh, "lspci | grep -iE 'nvidia|vga|3d' 2>/dev/null || echo 'lspci unavailable'")
run_raw(ssh, "ls /usr/local/ | grep -i cuda || echo 'No cuda in /usr/local'")
run_raw(ssh, "ls /dev/nvidia* 2>/dev/null || echo 'No /dev/nvidia devices'")
run_raw(ssh, "cat /proc/driver/nvidia/version 2>/dev/null || echo 'No NVIDIA kernel module loaded'")
run_raw(ssh, "dpkg -l | grep -i 'nvidia-driver' | head -5 || echo 'No nvidia driver packages found'")

# ── 2. Miniconda Installation ─────────────────────────────────────────────────
print("\n" + "="*55)
print("2. Miniconda Installation")
print("="*55)
already = run_raw(ssh, f"test -f {CONDA} && echo FOUND || echo NOTFOUND", show=False)
if "FOUND" in already:
    print(f"  ✅ Miniconda already at {CONDA}")
    run_raw(ssh, f"{CONDA} --version")
else:
    print("  Downloading Miniconda3...")
    run_raw(ssh, "wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && echo 'Downloaded'", timeout=120)
    print("  Installing (this takes ~1 min)...")
    run_raw(ssh, "bash /tmp/miniconda.sh -b -p /home/areeba/miniconda3 2>&1 | tail -5", timeout=300)
    ver = run_raw(ssh, f"{CONDA} --version", show=False)
    if "conda" in ver:
        print(f"  ✅ Miniconda installed: {ver}")
    else:
        print("  ❌ Miniconda install failed!")
        ssh.close(); sys.exit(1)

# ── 3. Create orbsfm Conda Environment ───────────────────────────────────────
print("\n" + "="*55)
print("3. Create orbsfm Environment")
print("="*55)
envs = run_raw(ssh, f"{CONDA} env list 2>/dev/null", show=False)
if "orbsfm" in envs:
    print("  ✅ orbsfm environment already exists")
else:
    print("  Creating from environment.yaml (2-5 mins)...")
    out = run_raw(ssh, f"{CONDA} env create -f {REMOTE_DIR}/environment.yaml 2>&1 | tail -8", timeout=600)
    if "orbsfm" in run_raw(ssh, f"{CONDA} env list", show=False):
        print("  ✅ Environment created!")
    else:
        print("  ⚠️  env create may have failed, trying to continue anyway")

# ── 4. Pip Dependencies ───────────────────────────────────────────────────────
print("\n" + "="*55)
print("4. Installing Pip Dependencies")
print("="*55)
PIP = f"{CONDA} run -n orbsfm pip install"
pkgs = ("pytorch-lightning==1.9.5 matplotlib opencv-python tqdm imageio "
        "path scipy configargparse kornia einops blessings progressbar2 "
        "'protobuf==4.25.6' scikit-image scikit-learn gdown")
run_raw(ssh, f"{PIP} {pkgs} 2>&1 | tail -5", timeout=300)

# ── 5. Detect CUDA & Install Correct PyTorch ─────────────────────────────────
print("\n" + "="*55)
print("5. Detect CUDA & Install PyTorch")
print("="*55)
cuda_check = run_raw(ssh, """
for d in /usr/local/cuda /usr/local/cuda-*; do
  if [ -f "$d/version.txt" ]; then cat "$d/version.txt"; break; fi
  if [ -f "$d/version.json" ]; then grep '"version"' "$d/version.json" | head -1; break; fi
done
nvcc --version 2>/dev/null | grep release || true
ls /usr/local/ | grep cuda || true
""", show=True)

# Parse CUDA version
cuda_ver = "unknown"
for line in cuda_check.split('\n'):
    import re
    m = re.search(r'(\d+\.\d+)', line)
    if m and 'cuda' in line.lower() or 'release' in line.lower() or 'version' in line.lower():
        cuda_ver = m.group(1)
        break

print(f"\n  Parsed CUDA version: {cuda_ver}")

if cuda_ver.startswith("11.7"):
    torch_cmd = f'{PIP} "torch==1.13.1+cu117" "torchvision==0.14.1+cu117" --index-url https://download.pytorch.org/whl/cu117'
elif cuda_ver.startswith("11.8"):
    torch_cmd = f'{PIP} "torch==2.0.0+cu118" "torchvision==0.15.0+cu118" --index-url https://download.pytorch.org/whl/cu118'
elif cuda_ver.startswith("12"):
    torch_cmd = f'{PIP} "torch==2.1.0+cu121" "torchvision==0.16.0+cu121" --index-url https://download.pytorch.org/whl/cu121'
else:
    print("  ⚠️  CUDA version unknown — will install cu117 as default")
    torch_cmd = f'{PIP} "torch==1.13.1+cu117" "torchvision==0.14.1+cu117" --index-url https://download.pytorch.org/whl/cu117'

print(f"  Installing: {torch_cmd.split('install')[1][:60]}...")
run_raw(ssh, f"{torch_cmd} 2>&1 | tail -5", timeout=300)

# ── 6. Verify PyTorch + CUDA ─────────────────────────────────────────────────
print("\n" + "="*55)
print("6. Verification")
print("="*55)
run_raw(ssh, f"""{CONDA} run -n orbsfm python -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
" 2>&1""")

# ── 7. KITTI Download Status ─────────────────────────────────────────────────
print("\n" + "="*55)
print("7. KITTI Download Status")
print("="*55)
run_raw(ssh, f"ls -lh {REMOTE_DIR}/data/data_odometry_color.zip 2>/dev/null | awk '{{print \"Downloaded:\", $5}}' || echo 'Not started yet'")
run_raw(ssh, f"tail -3 {REMOTE_DIR}/data/kitti_download.log 2>/dev/null || echo 'No download log'")
# Re-start download if not running
pid_check = run_raw(ssh, "ps aux | grep 'data_odometry_color' | grep -v grep | awk '{print $2}'", show=False)
if pid_check.strip():
    print(f"  ✅ Download running (PID: {pid_check.strip()})")
else:
    print("  Starting KITTI download...")
    run_raw(ssh, f"nohup wget -c 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip' -O {REMOTE_DIR}/data/data_odometry_color.zip >> {REMOTE_DIR}/data/kitti_download.log 2>&1 & echo 'Started PID:' $!")

ssh.close()
print("\n" + "="*55)
print("✅ Remote setup complete!")
print("="*55)
print(f"""
Next steps (after KITTI download finishes ~65GB):
  ssh areeba@{HOST}
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate orbsfm
  cd ~/SG-VO
  bash scripts/setup_dataset.sh   # extract zip + place cam.txt
  bash scripts/test_kitti_vo.sh   # run VO evaluation
""")
