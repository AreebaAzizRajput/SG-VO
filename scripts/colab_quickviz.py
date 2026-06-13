
# ============================================================
# SG-VO QUICK VISUALIZER — No reloading needed
# Assumes models + data already loaded from earlier cells.
# Runs inference-only (no adaptation) for fast visualization.
# ============================================================

import sys, os, traceback, copy
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio_v2
from imageio.v2 import imread
from skimage.transform import resize as imresize
from skimage import img_as_ubyte
from path import Path
from IPython.display import clear_output, display
import matplotlib.patches as mpatches

# ── CONFIG (edit these) ───────────────────────────────────────────────────────
SEQUENCES     = ['09', '10']          # which sequences to visualize
DATASET_DIR   = '/content/SG-VO/data/kitti_odom/sequences/'
POSE_NET_PATH = '/content/SG-VO/checkpoints/exp_pose112_model_best.pth.tar'
DISP_NET_PATH = '/content/SG-VO/checkpoints/dispnet112_model_best.pth.tar'
IMG_H, IMG_W  = 256, 832
RESNET_LAYERS = 50
SAMPLE_EVERY  = 20    # only infer every Nth frame (speeds up viz, set to 1 for all)
# ─────────────────────────────────────────────────────────────────────────────

# ── Check / recover models from current namespace ─────────────────────────────
sys.path.insert(0, '/content/SG-VO')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def _check_or_load(var_name, loader_fn):
    val = globals().get(var_name) or __builtins__.__dict__.get(var_name) if hasattr(__builtins__, '__dict__') else None
    # Try the notebook global scope via IPython
    try:
        import IPython
        shell = IPython.get_ipython()
        if shell and var_name in shell.user_ns:
            print(f"  ✅ {var_name} found in session — reusing.")
            return shell.user_ns[var_name]
    except Exception:
        pass
    print(f"  ⚠️  {var_name} not found — loading fresh...")
    return loader_fn()

print("Checking session for existing models...")

from models import DispResNet, PoseResNet
from utils import compute_depth, compute_pose_with_inv, extract_orb, tensor2array
from inverse_warp import pose_vec2mat

def _load_pose():
    w = torch.load(POSE_NET_PATH, map_location=device)
    m = PoseResNet().to(device)
    m.load_state_dict(w['state_dict'], strict=True)
    m.eval()
    return m

def _load_disp():
    w = torch.load(DISP_NET_PATH, map_location=device)
    m = DispResNet(RESNET_LAYERS, False).to(device)
    m.load_state_dict(w['state_dict'], strict=True)
    m.eval()
    return m

_pose_net = _check_or_load('pose_net', _load_pose)
_disp_net = _check_or_load('disp_net', _load_disp)
_pose_net.eval(); _disp_net.eval()
print(f"Device: {device}\n")

# ── Helper functions ──────────────────────────────────────────────────────────
def load_tensor(path):
    img = imread(path)
    if img.shape[0] != IMG_H or img.shape[1] != IMG_W:
        img = img_as_ubyte(imresize(img, (IMG_H, IMG_W)))
    img = extract_orb(img).astype(np.float32)
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0) / 255.0
    t[:, :3] = (t[:, :3] - 0.45) / 0.225
    return t

def denorm(tensor):
    t = tensor.squeeze()[:3].cpu().numpy().transpose(1, 2, 0)
    return np.clip(t * 0.225 + 0.45, 0, 1)

def colorize(disp):
    d = disp.squeeze().cpu().numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return plt.cm.inferno(d)[:, :, :3]

def pose_to_mat(pose_vec):
    mat = pose_vec2mat(pose_vec).squeeze(0).cpu().detach().numpy()
    return np.vstack([mat, [0, 0, 0, 1]])

# ── Run inference on each sequence ───────────────────────────────────────────
SEQ_COLORS    = ['#58a6ff', '#f78166']
all_results   = {}

for seq_idx, seq in enumerate(SEQUENCES):
    print(f"{'='*60}")
    print(f"Processing sequence {seq}...")

    image_dir  = Path(DATASET_DIR + seq + '/image_2/')
    files      = sorted(image_dir.files('*.png') + image_dir.files('*.jpg'))
    n          = len(files)

    if n == 0:
        print(f"  ❌ No images found in {image_dir} — skipping.")
        continue
    print(f"  Found {n} images")

    # Load intrinsics
    cam_txt = image_dir / 'cam.txt'
    if not cam_txt.exists():
        print(f"  ❌ cam.txt missing at {cam_txt} — skipping.")
        continue
    K = np.genfromtxt(cam_txt).astype(np.float32).reshape(3, 3)
    K_t = torch.from_numpy(K).unsqueeze(0).to(device)
    print(f"  Intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

    # Inference loop
    global_pose = np.eye(4)
    trajectory  = [global_pose[:3, :].reshape(1, 12)]
    sample_imgs = []   # (frame_idx, tgt_rgb, depth_rgb)
    errors      = []

    indices = list(range(0, n - 1))

    for i in indices:
        try:
            tgt = load_tensor(files[i]).to(device)
            ref = load_tensor(files[i + 1]).to(device)

            with torch.no_grad():
                pose_vec  = _pose_net(tgt, ref)
                disp_pred = _disp_net(tgt[:, :3])[0]

            mat = pose_to_mat(pose_vec)
            global_pose = global_pose @ np.linalg.inv(mat)
            trajectory.append(global_pose[:3, :].reshape(1, 12))

            # Collect sample frames every SAMPLE_EVERY frames
            if i % SAMPLE_EVERY == 0:
                sample_imgs.append((i, denorm(tgt), colorize(disp_pred)))

        except Exception as e:
            errors.append((i, str(e)))
            if len(errors) == 1:
                print(f"  ⚠️  First error at frame {i}: {e}")
                traceback.print_exc()
            continue

    traj_np = np.concatenate(trajectory, axis=0)
    all_results[seq] = {
        'traj':    traj_np,
        'samples': sample_imgs,
        'errors':  errors,
        'n':       n,
        'K':       K,
    }

    # Save trajectory
    out_path = f'/content/SG-VO/vo_results_online/{seq}_quickviz.txt'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, traj_np, delimiter=' ', fmt='%1.8e')

    ok = len(indices) - len(errors)
    print(f"  ✅ Done: {ok}/{len(indices)} frames OK, {len(errors)} errors")
    print(f"  Trajectory saved: {out_path}")

# ── Master Visualization ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Generating visualization...")

n_seqs = len(all_results)
if n_seqs == 0:
    print("No sequences processed — nothing to plot.")
else:
    # --- Part 1: Depth sample grid per sequence ---
    for seq, res in all_results.items():
        samples = res['samples']
        if not samples:
            continue
        cols = min(4, len(samples))
        rows = len(samples) // cols + (1 if len(samples) % cols else 0)

        fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 5, rows * 4))
        fig.patch.set_facecolor('#0d1117')
        fig.suptitle(f'Sequence {seq} — RGB (top) & Predicted Depth (bottom)',
                     color='white', fontsize=14, fontweight='bold', y=1.01)

        for flat_i, (frame_idx, rgb, depth) in enumerate(samples):
            r, c = flat_i // cols, flat_i % cols
            for ax_row, img, title in [
                (r * 2,     rgb,   f'Frame {frame_idx}'),
                (r * 2 + 1, depth, f'Depth {frame_idx}'),
            ]:
                if rows * 2 == 1:
                    ax = axes[c] if cols > 1 else axes
                else:
                    ax = axes[ax_row, c] if cols > 1 else axes[ax_row]
                ax.imshow(img)
                ax.set_title(title, color='white', fontsize=9)
                ax.axis('off')

        # Hide unused subplots
        total_slots = rows * cols
        for extra in range(len(samples), total_slots):
            r, c = extra // cols, extra % cols
            for ax_row in [r * 2, r * 2 + 1]:
                try:
                    ax = axes[ax_row, c] if cols > 1 else axes[ax_row]
                    ax.axis('off')
                except Exception:
                    pass

        plt.tight_layout()
        plt.show()

    # --- Part 2: Trajectories side-by-side ---
    fig, axes = plt.subplots(1, n_seqs, figsize=(12 * n_seqs, 8))
    fig.patch.set_facecolor('#0d1117')
    if n_seqs == 1:
        axes = [axes]

    for ax, (seq, res), col in zip(axes, all_results.items(), SEQ_COLORS):
        traj = res['traj']
        xs   = traj[:, 3]
        zs   = traj[:, 11]
        ax.set_facecolor('#161b22')
        sc = ax.scatter(xs, zs, c=np.arange(len(xs)), cmap='plasma', s=8, zorder=3)
        ax.plot(xs, zs, color=col, linewidth=0.9, alpha=0.6)
        ax.scatter([xs[0]],  [zs[0]],  c='lime',  s=120, zorder=6, label='Start')
        ax.scatter([xs[-1]], [zs[-1]], c='red',   s=120, zorder=6, label='End')

        plt.colorbar(sc, ax=ax).set_label('Frame index', color='white')
        ax.set_title(f'Sequence {seq} Trajectory (X-Z)', color='white',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('X (m)', color='white'); ax.set_ylabel('Z (m)', color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#21262d', labelcolor='white')
        ax.set_aspect('equal')
        ax.grid(True, color='#30363d', linewidth=0.5)
        for sp in ax.spines.values(): sp.set_color('#30363d')

        # Annotate stats
        total_dist = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(zs)**2))
        n_err = len(res['errors'])
        ax.text(0.02, 0.98,
                f"Frames: {res['n']}  |  Errors: {n_err}\n"
                f"Path length: {total_dist:.1f}m",
                transform=ax.transAxes, color='white', fontsize=10,
                va='top', bbox=dict(boxstyle='round', fc='#21262d', ec='#30363d'))

    fig.suptitle('SG-VO Trajectory Comparison', color='white',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # --- Part 3: Error summary ---
    print("\n" + "="*60)
    print("ERROR SUMMARY")
    print("="*60)
    for seq, res in all_results.items():
        errs = res['errors']
        print(f"Seq {seq}: {len(errs)} errors out of {res['n']-1} frames")
        for frame_i, msg in errs[:5]:
            print(f"  Frame {frame_i:4d}: {msg}")
        if len(errs) > 5:
            print(f"  ... and {len(errs)-5} more")

    print("\nAll done! Trajectories saved to /content/SG-VO/vo_results_online/")
