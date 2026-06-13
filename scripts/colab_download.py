import os, zipfile, shutil, sys, traceback
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as _iio
from skimage.transform import resize as _imresize
from skimage import img_as_ubyte as _ub
from path import Path as _Path
from google.colab import files

SAVE_DIR    = '/content/SG-VO/viz_export'
RESULTS_DIR = '/content/SG-VO/vo_results_online'
os.makedirs(SAVE_DIR, exist_ok=True)

TRAJ_FILES = {
    '09': os.path.join(RESULTS_DIR, '09_quickviz.txt'),
    '10': os.path.join(RESULTS_DIR, '10_quickviz.txt'),
}
SEQ_COLORS = {'09': '#58a6ff', '10': '#f78166'}

# ── 1. Individual trajectory plots ───────────────────────────────────────────
for seq, fpath in TRAJ_FILES.items():
    if not os.path.exists(fpath):
        print(f"Missing: {fpath}"); continue
    traj = np.loadtxt(fpath)
    xs, zs = traj[:, 3], traj[:, 11]
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
    sc = ax.scatter(xs, zs, c=np.arange(len(xs)), cmap='plasma', s=10, zorder=3)
    ax.plot(xs, zs, color=SEQ_COLORS[seq], linewidth=1.2, alpha=0.6)
    ax.scatter([xs[0]], [zs[0]], c='lime', s=150, zorder=6, label='Start')
    ax.scatter([xs[-1]], [zs[-1]], c='red', s=150, zorder=6, label='End')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Frame index', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    dist = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(zs)**2))
    ax.text(0.02, 0.97, f"Frames: {len(traj)}  |  Path: {dist:.1f} m",
            transform=ax.transAxes, color='white', fontsize=11, va='top',
            bbox=dict(boxstyle='round', fc='#21262d', ec='#30363d'))
    ax.set_title(f'SG-VO Trajectory — Sequence {seq}', color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)', color='white'); ax.set_ylabel('Z (m)', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#21262d', labelcolor='white', fontsize=10)
    ax.set_aspect('equal'); ax.grid(True, color='#30363d', linewidth=0.5)
    for sp in ax.spines.values(): sp.set_color('#30363d')
    out = os.path.join(SAVE_DIR, f'trajectory_seq{seq}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig); print(f"Saved: {out}")

# ── 2. Side-by-side comparison ───────────────────────────────────────────────
trajs = {seq: np.loadtxt(fp) for seq, fp in TRAJ_FILES.items() if os.path.exists(fp)}
if len(trajs) == 2:
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.patch.set_facecolor('#0d1117')
    for ax, (seq, traj) in zip(axes, trajs.items()):
        xs, zs = traj[:, 3], traj[:, 11]
        ax.set_facecolor('#161b22')
        sc = ax.scatter(xs, zs, c=np.arange(len(xs)), cmap='plasma', s=8, zorder=3)
        ax.plot(xs, zs, color=SEQ_COLORS[seq], linewidth=1.0, alpha=0.6)
        ax.scatter([xs[0]], [zs[0]], c='lime', s=120, zorder=6, label='Start')
        ax.scatter([xs[-1]], [zs[-1]], c='red', s=120, zorder=6, label='End')
        dist = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(zs)**2))
        ax.set_title(f'Sequence {seq}  ({len(traj)} frames, {dist:.0f} m)',
                     color='white', fontsize=13, fontweight='bold')
        ax.set_xlabel('X (m)', color='white'); ax.set_ylabel('Z (m)', color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#21262d', labelcolor='white')
        ax.set_aspect('equal'); ax.grid(True, color='#30363d', linewidth=0.5)
        for sp in ax.spines.values(): sp.set_color('#30363d')
        plt.colorbar(sc, ax=ax).set_label('Frame', color='white')
    fig.suptitle('SG-VO — Trajectory Comparison (Seq 09 vs 10)',
                 color='white', fontsize=16, fontweight='bold')
    out = os.path.join(SAVE_DIR, 'trajectory_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig); print(f"Saved: {out}")

# ── 3. Copy raw pose txt files ────────────────────────────────────────────────
for seq, fpath in TRAJ_FILES.items():
    if os.path.exists(fpath):
        dest = os.path.join(SAVE_DIR, f'poses_seq{seq}.txt')
        shutil.copy(fpath, dest); print(f"Copied: {dest}")

# ── 4. Depth + Error grids ───────────────────────────────────────────────────
sys.path.insert(0, '/content/SG-VO')

IMG_H_D, IMG_W_D   = 256, 832
DEPTH_SAMPLE_EVERY = 30
_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def _raw_load(path):
    img = _iio.imread(path)
    if img.shape[0] != IMG_H_D or img.shape[1] != IMG_W_D:
        img = _ub(_imresize(img, (IMG_H_D, IMG_W_D)))
    return img

def _to_tensor(img, extract_fn):
    img = extract_fn(img).astype('float32')
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0) / 255.0
    t[:, :3] = (t[:, :3] - 0.45) / 0.225
    return t

def _denorm(tensor):
    t = tensor.squeeze()[:3].cpu().numpy().transpose(1, 2, 0)
    return np.clip(t * 0.225 + 0.45, 0, 1)

try:
    from utils import extract_orb as _orb
    from models import DispResNet as _Disp, PoseResNet as _Pose
    from inverse_warp import inverse_warp

    _dnet = _Disp(50, False).to(_device)
    _wd = torch.load('/content/SG-VO/checkpoints/dispnet112_model_best.pth.tar', map_location=_device)
    _dnet.load_state_dict(_wd['state_dict'], strict=True); _dnet.eval()

    _pnet = _Pose().to(_device)
    _wp = torch.load('/content/SG-VO/checkpoints/exp_pose112_model_best.pth.tar', map_location=_device)
    _pnet.load_state_dict(_wp['state_dict'], strict=True); _pnet.eval()
    print("\n[Depth + Error grids] Models loaded.")

    for seq in ['09', '10']:
        img_dir   = _Path(f'/content/SG-VO/data/kitti_odom/sequences/{seq}/image_2/')
        img_files = sorted(img_dir.files('*.png') + img_dir.files('*.jpg'))
        K = np.genfromtxt(img_dir / 'cam.txt').astype(np.float32).reshape(3, 3)
        K_t = torch.from_numpy(K).unsqueeze(0).to(_device)

        indices = list(range(0, len(img_files) - 1, DEPTH_SAMPLE_EVERY))
        samples = []  # (fidx, rgb, depth_col, err_col, mean_err, max_err)

        print(f"  Seq {seq}: {len(indices)} samples (RGB / Depth / Error)...")
        for idx in indices:
            try:
                raw_tgt = _raw_load(img_files[idx])
                raw_ref = _raw_load(img_files[idx + 1])
                tgt_t = _to_tensor(raw_tgt, _orb).to(_device)
                ref_t = _to_tensor(raw_ref, _orb).to(_device)

                with torch.no_grad():
                    disp     = _dnet(tgt_t[:, :3])[0]
                    # Normalise to (B, 1, H, W) regardless of model output shape
                    if disp.dim() == 3:   # (B, H, W) — add channel dim
                        disp = disp.unsqueeze(1)
                    elif disp.dim() == 2: # (H, W) — add batch + channel
                        disp = disp.unsqueeze(0).unsqueeze(0)
                    depth     = 1.0 / disp.clamp(min=1e-3)   # (B, 1, H, W)
                    depth_bhw = depth.squeeze(1)              # (B, H, W) for inverse_warp
                    pose_vec  = _pnet(tgt_t, ref_t)
                    warped, valid_mask = inverse_warp(
                        ref_t[:, :3], depth_bhw, pose_vec, K_t,
                        rotation_mode='euler', padding_mode='zeros')

                # Photometric error (L1, masked)
                err_map = (tgt_t[:, :3] - warped).abs()
                err_map = err_map * valid_mask.unsqueeze(1).float()
                err_hw  = err_map.squeeze(0).mean(0).cpu().numpy()
                valid   = err_hw > 0
                mean_err = float(err_hw[valid].mean()) if valid.any() else 0.0
                max_err  = float(err_hw.max())

                err_col   = plt.cm.jet(err_hw / (max_err + 1e-6))[:, :, :3]
                d_np      = disp.squeeze().cpu().numpy()
                depth_col = plt.cm.inferno(
                    (d_np - d_np.min()) / (d_np.max() - d_np.min() + 1e-6))[:, :, :3]
                rgb_show  = _denorm(tgt_t)

                samples.append((idx, rgb_show, depth_col, err_col, mean_err, max_err))
            except Exception as e:
                print(f"    Frame {idx} skipped: {e}")

        if not samples:
            print(f"  No samples for seq {seq}"); continue

        # ── 3-row grid: RGB / Depth / Error ──────────────────────────────────
        cols  = min(5, len(samples))
        nrows = (len(samples) + cols - 1) // cols
        fig = plt.figure(figsize=(cols * 4.5, nrows * 3 * 2.8))
        fig.patch.set_facecolor('#0d1117')
        fig.suptitle(
            f'Seq {seq}  |  Row 1: RGB  |  Row 2: Depth (inferno)  |  Row 3: Reprojection Error (jet)',
            color='white', fontsize=11, fontweight='bold', y=1.005)
        outer = fig.add_gridspec(nrows, cols, hspace=0.06, wspace=0.04)

        for flat_i, (fidx, rgb, depth_col, err_col, mean_err, max_err) in enumerate(samples):
            rg, ci = flat_i // cols, flat_i % cols
            inner  = outer[rg, ci].subgridspec(3, 1, hspace=0.04)
            rows_data = [
                (rgb,       f'#{fidx} RGB', False),
                (depth_col, f'#{fidx} Depth', False),
                (err_col,   f'#{fidx} Err μ={mean_err:.3f} max={max_err:.3f}', mean_err > 0.15),
            ]
            for sub_r, (img, lbl, highlight) in enumerate(rows_data):
                ax = fig.add_subplot(inner[sub_r])
                ax.imshow(img, aspect='auto')
                ax.set_title(lbl, color='#f78166' if highlight else 'white',
                             fontsize=7, pad=2)
                ax.axis('off')
                if highlight:
                    for sp in ax.spines.values():
                        sp.set_visible(True); sp.set_color('#f78166'); sp.set_linewidth(2.5)

        out = os.path.join(SAVE_DIR, f'depth_error_grid_seq{seq}.png')
        fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='#0d1117')
        plt.close(fig); print(f"  Saved: {out}")

        # ── Error-over-time plot ──────────────────────────────────────────────
        mean_errs = [s[4] for s in samples]
        frame_ids = [s[0] for s in samples]
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        fig2.patch.set_facecolor('#0d1117'); ax2.set_facecolor('#161b22')
        ax2.plot(frame_ids, mean_errs, color='#f78166', linewidth=1.5,
                 marker='o', markersize=4, label='Mean reprojection error')
        ax2.axhline(np.mean(mean_errs), color='#ffa657', linestyle='--', linewidth=1.2,
                    label=f'Average = {np.mean(mean_errs):.4f}')
        ax2.fill_between(frame_ids, mean_errs, alpha=0.2, color='#f78166')
        threshold = np.mean(mean_errs) + np.std(mean_errs)
        spikes = [(f, e) for f, e in zip(frame_ids, mean_errs) if e > threshold]
        if spikes:
            sx, sy = zip(*spikes)
            ax2.scatter(sx, sy, c='red', s=60, zorder=5, label='High-error frames')
        ax2.set_title(f'Sequence {seq} — Per-Frame Reprojection Error Over Time',
                      color='white', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Frame index', color='white'); ax2.set_ylabel('Mean L1 error', color='white')
        ax2.tick_params(colors='white')
        ax2.legend(facecolor='#21262d', labelcolor='white', fontsize=9)
        ax2.grid(True, color='#30363d', linewidth=0.5)
        for sp in ax2.spines.values(): sp.set_color('#30363d')
        out2 = os.path.join(SAVE_DIR, f'error_over_time_seq{seq}.png')
        fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close(fig2); print(f"  Saved: {out2}")

except Exception as e:
    print(f"[Depth + Error grids] Failed: {e}")
    traceback.print_exc()

# ── 5. Zip and download ───────────────────────────────────────────────────────
zip_path = '/content/sgvo_results.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(os.listdir(SAVE_DIR)):
        full = os.path.join(SAVE_DIR, fname)
        zf.write(full, fname)
        print(f"  + {fname}")

size_mb = os.path.getsize(zip_path) / 1e6
print(f"\nZip ready: {zip_path}  ({size_mb:.1f} MB)")
print("Starting download...")
files.download(zip_path)
