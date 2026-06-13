#!/usr/bin/env python3
"""
Remote GPU server setup script for SG-VO.
Tests connection, transfers files, and sets up the environment.
"""
import paramiko
import os
import sys
import tarfile
import io
import time

HOST = "119.63.132.182"
USER = "areeba"
PASS = "51gm@123++"
REPO_LOCAL = "/home/areeba/ICRAMaxxing/SG-VO"
REMOTE_DIR = "/home/areeba/SG-VO"

def run_remote(ssh, cmd, timeout=60, print_output=True):
    print(f"\n$ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out and print_output:
        print(out.rstrip())
    if err and print_output:
        print("[STDERR]", err.rstrip())
    return out, err

def connect():
    print(f"Connecting to {USER}@{HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=15)
    print("✅ Connected!")
    return ssh

def check_remote(ssh):
    print("\n=== Remote Machine Info ===")
    run_remote(ssh, "echo 'User:' $(whoami) && uname -r")
    run_remote(ssh, "nvidia-smi | head -15 || echo '⚠️  nvidia-smi not found'")
    run_remote(ssh, "conda --version 2>/dev/null || echo '⚠️  conda not found'")
    run_remote(ssh, "df -h ~ | tail -1")

def setup_remote_dirs(ssh):
    print("\n=== Creating Remote Directories ===")
    dirs = [
        f"{REMOTE_DIR}/checkpoints",
        f"{REMOTE_DIR}/data/kitti_odom/sequences",
        f"{REMOTE_DIR}/data/kitti_odom/sequences/kitti_odom256_intrinsics",
        f"{REMOTE_DIR}/data/kitti_256",
        f"{REMOTE_DIR}/vo_results",
        f"{REMOTE_DIR}/vo_results_online",
    ]
    run_remote(ssh, "mkdir -p " + " ".join(dirs))
    print("✅ Directories created")

def upload_files(ssh):
    sftp = ssh.open_sftp()
    print("\n=== Uploading Files ===")

    # Files/dirs to transfer
    transfers = [
        # (local_path, remote_path)
        # Code files
        (f"{REPO_LOCAL}/train.py", f"{REMOTE_DIR}/train.py"),
        (f"{REPO_LOCAL}/test_vo.py", f"{REMOTE_DIR}/test_vo.py"),
        (f"{REPO_LOCAL}/test_vo_online.py", f"{REMOTE_DIR}/test_vo_online.py"),
        (f"{REPO_LOCAL}/run_inference.py", f"{REMOTE_DIR}/run_inference.py"),
        (f"{REPO_LOCAL}/inverse_warp.py", f"{REMOTE_DIR}/inverse_warp.py"),
        (f"{REPO_LOCAL}/loss_functions.py", f"{REMOTE_DIR}/loss_functions.py"),
        (f"{REPO_LOCAL}/utils.py", f"{REMOTE_DIR}/utils.py"),
        (f"{REPO_LOCAL}/logger.py", f"{REMOTE_DIR}/logger.py"),
        (f"{REPO_LOCAL}/custom_transforms.py", f"{REMOTE_DIR}/custom_transforms.py"),
        (f"{REPO_LOCAL}/visualize_attention.py", f"{REMOTE_DIR}/visualize_attention.py"),
        (f"{REPO_LOCAL}/environment.yaml", f"{REMOTE_DIR}/environment.yaml"),
        # Scripts
        (f"{REPO_LOCAL}/scripts/train.sh", f"{REMOTE_DIR}/scripts/train.sh"),
        (f"{REPO_LOCAL}/scripts/test_kitti_vo.sh", f"{REMOTE_DIR}/scripts/test_kitti_vo.sh"),
        (f"{REPO_LOCAL}/scripts/test_kitti_vo_online.sh", f"{REMOTE_DIR}/scripts/test_kitti_vo_online.sh"),
        (f"{REPO_LOCAL}/scripts/setup_dataset.sh", f"{REMOTE_DIR}/scripts/setup_dataset.sh"),
        # Checkpoints
        (f"{REPO_LOCAL}/checkpoints/dispnet112_model_best.pth.tar", f"{REMOTE_DIR}/checkpoints/dispnet112_model_best.pth.tar"),
        (f"{REPO_LOCAL}/checkpoints/exp_pose112_model_best.pth.tar", f"{REMOTE_DIR}/checkpoints/exp_pose112_model_best.pth.tar"),
    ]

    # Create remote script dir
    try: sftp.mkdir(f"{REMOTE_DIR}/scripts")
    except: pass

    for local, remote in transfers:
        if not os.path.exists(local):
            print(f"  ⚠️  Skipping (not found): {local}")
            continue
        size_mb = os.path.getsize(local) / (1024*1024)
        print(f"  Uploading {os.path.basename(local)} ({size_mb:.1f} MB)...", end="", flush=True)
        sftp.put(local, remote)
        print(" ✅")

    # Upload directories (datasets, models, kitti_eval)
    for subdir in ["datasets", "models", "kitti_eval", "data/kitti_odom/sequences/kitti_odom256_intrinsics"]:
        local_dir = f"{REPO_LOCAL}/{subdir}"
        remote_dir = f"{REMOTE_DIR}/{subdir}"
        if not os.path.exists(local_dir):
            continue
        try:
            sftp.mkdir(remote_dir)
        except:
            pass
        for fname in os.listdir(local_dir):
            local_f = f"{local_dir}/{fname}"
            remote_f = f"{remote_dir}/{fname}"
            if os.path.isfile(local_f):
                size_mb = os.path.getsize(local_f) / (1024*1024)
                print(f"  Uploading {subdir}/{fname} ({size_mb:.1f} MB)...", end="", flush=True)
                sftp.put(local_f, remote_f)
                print(" ✅")

    sftp.close()
    print("\n✅ All files uploaded!")

def update_remote_scripts(ssh):
    """Update paths in remote scripts to point to REMOTE_DIR"""
    print("\n=== Updating Remote Script Paths ===")
    sed_cmds = [
        f"sed -i 's|/home/areeba/ICRAMaxxing/SG-VO|{REMOTE_DIR}|g' {REMOTE_DIR}/scripts/train.sh",
        f"sed -i 's|/home/areeba/ICRAMaxxing/SG-VO|{REMOTE_DIR}|g' {REMOTE_DIR}/scripts/test_kitti_vo.sh",
        f"sed -i 's|/home/areeba/ICRAMaxxing/SG-VO|{REMOTE_DIR}|g' {REMOTE_DIR}/scripts/test_kitti_vo_online.sh",
        f"sed -i 's|/home/areeba/ICRAMaxxing/SG-VO|{REMOTE_DIR}|g' {REMOTE_DIR}/scripts/setup_dataset.sh",
        f"chmod +x {REMOTE_DIR}/scripts/*.sh",
    ]
    for cmd in sed_cmds:
        run_remote(ssh, cmd)
    print("✅ Script paths updated")

def setup_conda_env(ssh):
    print("\n=== Setting Up Conda Environment ===")
    # Check if orbsfm env already exists
    out, _ = run_remote(ssh, "conda env list 2>/dev/null | grep orbsfm || echo 'NOT_FOUND'")
    if "orbsfm" in out and "NOT_FOUND" not in out:
        print("✅ orbsfm environment already exists")
    else:
        print("Creating orbsfm conda environment (this takes a few minutes)...")
        run_remote(ssh, f"conda env create -f {REMOTE_DIR}/environment.yaml 2>&1 | tail -5", timeout=600)

    # Install pip deps
    print("\nInstalling pip dependencies...")
    pip_cmd = (
        f"conda run -n orbsfm pip install "
        f"pytorch-lightning==1.9.5 matplotlib opencv-python tqdm imageio "
        f"path scipy configargparse kornia einops blessings progressbar2 "
        f"'protobuf==4.25.6' scikit-image scikit-learn gdown paramiko 2>&1 | tail -5"
    )
    run_remote(ssh, pip_cmd, timeout=300)

def check_cuda_on_remote(ssh):
    print("\n=== Verifying CUDA on Remote ===")
    out, _ = run_remote(ssh, "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'No GPU'")
    cuda_ver, _ = run_remote(ssh, "nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',' || nvidia-smi 2>/dev/null | grep 'CUDA Version' | awk '{print $9}' || echo 'unknown'")
    print(f"CUDA version: {cuda_ver.strip()}")

    # Install correct torch for detected CUDA
    run_remote(ssh, """
CUDA=$(nvidia-smi 2>/dev/null | grep 'CUDA Version' | awk '{print $9}' | cut -d. -f1-2)
echo "Detected CUDA: $CUDA"
if [[ "$CUDA" == "11.7" ]]; then
    conda run -n orbsfm pip install "torch==1.13.1+cu117" "torchvision==0.14.1+cu117" --index-url https://download.pytorch.org/whl/cu117 2>&1 | tail -3
elif [[ "$CUDA" == "11.8" ]]; then
    conda run -n orbsfm pip install "torch==2.0.0+cu118" "torchvision==0.15.0+cu118" --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -3
elif [[ "$CUDA" == "12.1" ]] || [[ "$CUDA" == "12.2" ]]; then
    conda run -n orbsfm pip install "torch==2.1.0+cu121" "torchvision==0.16.0+cu121" --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
else
    echo "CUDA $CUDA — installing torch for cu117 as default"
    conda run -n orbsfm pip install "torch==1.13.1+cu117" "torchvision==0.14.1+cu117" --index-url https://download.pytorch.org/whl/cu117 2>&1 | tail -3
fi
""", timeout=300)

def start_kitti_download(ssh):
    print("\n=== Starting KITTI Dataset Download on Remote ===")
    cmd = (
        f"nohup wget -c 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip' "
        f"-O {REMOTE_DIR}/data/data_odometry_color.zip "
        f">> {REMOTE_DIR}/data/kitti_download.log 2>&1 & echo \"Download started, PID: $!\""
    )
    run_remote(ssh, cmd)

def verify_remote(ssh):
    print("\n=== Final Verification ===")
    run_remote(ssh, f"ls {REMOTE_DIR}/checkpoints/*.pth.tar")
    run_remote(ssh, f"ls {REMOTE_DIR}/scripts/")
    run_remote(ssh, f"conda run -n orbsfm python -c \"import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())\" 2>&1")
    run_remote(ssh, f"ls {REMOTE_DIR}/data/")

def main():
    try:
        ssh = connect()
        check_remote(ssh)
        setup_remote_dirs(ssh)
        upload_files(ssh)
        update_remote_scripts(ssh)
        setup_conda_env(ssh)
        check_cuda_on_remote(ssh)
        start_kitti_download(ssh)
        verify_remote(ssh)

        print("\n" + "="*50)
        print("✅ REMOTE SETUP COMPLETE!")
        print("="*50)
        print(f"""
On the GPU server, once the KITTI download finishes:
  ssh areeba@{HOST}
  conda activate orbsfm
  cd {REMOTE_DIR}
  bash scripts/setup_dataset.sh   # extract + organize dataset
  bash scripts/test_kitti_vo.sh   # run evaluation
""")
        ssh.close()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
