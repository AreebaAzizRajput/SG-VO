"""
SG-VO Spatial Error Visualizer
Shows reprojection error overlaid directly on RGB frames.
High-error regions get red bounding boxes so you can see WHAT is failing.
Run: exec(open('/content/SG-VO/scripts/colab_spatial_error.py').read())
"""

import os, sys, traceback
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import label as ndlabel
import imageio.v2 as _iio
from skimage.transform import resize as _imresize
from skimage import img_as_ubyte as _ub
from path import Path as _Path
from zipfile import ZipFile
from google.colab import files

sys.path.insert(0, '/content/SG-VO')

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEQUENCES        = ['09', '10']
DATASET_DIR      = '/content/SG-VO/data/kitti_odom/sequences/'
DISP_CKPT        = '/content/SG-VO/checkpoints/dispnet112_model_best.pth.tar'
POSE_CKPT        = '/content/SG-VO/checkpoints/exp_pose112_model_best.pth.tar'
IMG_H, IMG_W     = 256, 832
SAMPLE_EVERY     = 50        # visualize 1 frame every N frames
TOP_K_BOXES      = 4         # how many high-error bounding boxes to draw
PATCH_SIZE       = 64        # size (px) of each patch analysed for box placement
ERROR_ALPHA      = 0.55      # opacity of error overlay (0=invisible, 1=opaque)
SAVE_DIR         = '/content/SG-VO/viz_export'
os.makedirs(SAVE_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def _raw_load(path):
    img = _iio.imread(path)
    if img.shape[:2] != (IMG_H, IMG_W):
        img = _ub(_imresize(img, (IMG_H, IMG_W)))
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    return img[:, :, :3]

def _to_tensor(img_rgb, extract_fn):
    arr = extract_fn(img_rgb).astype('float32')
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0) / 255.0
    t[:, :3] = (t[:, :3] - 0.45) / 0.225
    return t

def _top_k_error_boxes(err_hw, k=4, patch=64):
    """Find k non-overlapping patches with highest mean error. Returns list of (y0,x0,y1,x1)."""
    H, W = err_hw.shape
    rows = H // patch
    cols = W // patch
    scores = []
    for r in range(rows):
        for c in range(cols):
            patch_err = err_hw[r*patch:(r+1)*patch, c*patch:(c+1)*patch]
            scores.append((patch_err.mean(), r*patch, c*patch))
    scores.sort(reverse=True)
    boxes = []
    used  = set()
    for score, y0, x0 in scores:
        if len(boxes) >= k:
            break
        # Simple non-overlap check: avoid within 1 patch of an existing box
        occupied = False
        for by, bx in used:
            if abs(by - y0) < patch and abs(bx - x0) < patch:
                occupied = True; break
        if not occupied:
            boxes.append((y0, x0, min(y0+patch, H), min(x0+patch, W), score))
            used.add((y0, x0))
    return boxes

# ── Load models ───────────────────────────────────────────────────────────────
from utils import extract_orb as _orb
from models import DispResNet as _Disp, PoseResNet as _Pose
from inverse_warp import inverse_warp

_dnet = _Disp(50, False).to(_device)
_dnet.load_state_dict(torch.load(DISP_CKPT, map_location=_device)['state_dict'], strict=True)
_dnet.eval()

_pnet = _Pose().to(_device)
_pnet.load_state_dict(torch.load(POSE_CKPT, map_location=_device)['state_dict'], strict=True)
_pnet.eval()
print("Models loaded.")

# ── Main loop ─────────────────────────────────────────────────────────────────
saved_files = []

for seq in SEQUENCES:
    img_dir   = _Path(f'{DATASET_DIR}{seq}/image_2/')
    img_files = sorted(img_dir.files('*.png') + img_dir.files('*.jpg'))
    K = np.genfromtxt(img_dir / 'cam.txt').astype(np.float32).reshape(3, 3)
    K_t = torch.from_numpy(K).unsqueeze(0).to(_device)

    indices = list(range(0, len(img_files) - 1, SAMPLE_EVERY))
    print(f"\nSeq {seq}: {len(indices)} frames to visualize...")

    # Pre-generate all annotated frames
    frame_results = []

    for idx in indices:
        try:
            raw_tgt = _raw_load(img_files[idx])
            raw_ref = _raw_load(img_files[idx + 1])
            tgt_t   = _to_tensor(raw_tgt, _orb).to(_device)
            ref_t   = _to_tensor(raw_ref, _orb).to(_device)

            with torch.no_grad():
                disp = _dnet(tgt_t[:, :3])[0]
                if disp.dim() == 3:
                    disp = disp.unsqueeze(1)
                depth_bhw = (1.0 / disp.clamp(min=1e-3)).squeeze(1)
                pose_vec  = _pnet(tgt_t, ref_t)
                warped, valid_mask = inverse_warp(
                    ref_t[:, :3], depth_bhw, pose_vec, K_t,
                    rotation_mode='euler', padding_mode='zeros')

            # Reprojection error (H, W)
            err = (tgt_t[:, :3] - warped).abs()
            err = err * valid_mask.unsqueeze(1).float()
            err_hw = err.squeeze(0).mean(0).cpu().numpy()

            # Depth for display
            d = disp.squeeze().cpu().numpy()
            d = (d - d.min()) / (d.max() - d.min() + 1e-6)

            frame_results.append({
                'idx':    idx,
                'rgb':    raw_tgt,
                'depth':  d,
                'err_hw': err_hw,
                'mean_err': float(err_hw[err_hw > 0].mean()) if (err_hw > 0).any() else 0.0,
                'boxes':  _top_k_error_boxes(err_hw, k=TOP_K_BOXES, patch=PATCH_SIZE),
            })
        except Exception as e:
            print(f"  Frame {idx} skipped: {e}")

    if not frame_results:
        print(f"  No frames processed for seq {seq}")
        continue

    # ── Compute error percentile threshold for coloring boxes ────────────────
    all_scores = [b[4] for fr in frame_results for b in fr['boxes']]
    high_thresh = np.percentile(all_scores, 75) if all_scores else 0.2

    # ── Build figure grid ────────────────────────────────────────────────────
    # Each frame = 2 rows (annotated RGB, depth+pure error heatmap)
    COLS = min(4, len(frame_results))
    NROWS_PER_FRAME = 2
    n_groups = (len(frame_results) + COLS - 1) // COLS

    fig = plt.figure(figsize=(COLS * 5.5, n_groups * NROWS_PER_FRAME * 3.2))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        f'SG-VO Seq {seq} — Spatial Error Analysis\n'
        f'Row 1: RGB + error overlay + bounding boxes (red=critical, orange=moderate)\n'
        f'Row 2: Predicted depth  |  Pure error heatmap (jet: blue→red)',
        color='white', fontsize=11, fontweight='bold', y=1.01)

    outer = fig.add_gridspec(n_groups, COLS, hspace=0.08, wspace=0.04)

    for flat_i, fr in enumerate(frame_results):
        row_g = flat_i // COLS
        col_i = flat_i % COLS
        inner = outer[row_g, col_i].subgridspec(2, 2, hspace=0.03, wspace=0.03)

        rgb    = fr['rgb'].copy().astype(np.float32) / 255.0
        err_hw = fr['err_hw']
        depth  = fr['depth']
        boxes  = fr['boxes']

        # ── Top-left: RGB + semi-transparent error overlay + boxes ───────────
        ax_main = fig.add_subplot(inner[0, :])   # span both top columns

        # Error heatmap blended onto RGB
        err_norm = err_hw / (err_hw.max() + 1e-6)
        err_rgba = plt.cm.jet(err_norm)[:, :, :3]
        blended  = (1 - ERROR_ALPHA) * rgb + ERROR_ALPHA * err_rgba
        blended  = np.clip(blended, 0, 1)

        ax_main.imshow(blended, aspect='auto')
        ax_main.axis('off')

        # Draw bounding boxes
        for (y0, x0, y1, x1, score) in boxes:
            is_critical = score >= high_thresh
            ec = '#ff4444' if is_critical else '#ffa657'
            lw = 2.5 if is_critical else 1.8
            rect = mpatches.Rectangle(
                (x0, y0), x1-x0, y1-y0,
                linewidth=lw, edgecolor=ec, facecolor='none', zorder=5)
            ax_main.add_patch(rect)
            ax_main.text(x0+2, y0-3, f'{score:.3f}',
                         color=ec, fontsize=6, fontweight='bold',
                         bbox=dict(fc='#0d1117', ec='none', alpha=0.7, pad=0.5))

        # Frame label and mean error badge
        badge_col = '#ff4444' if fr['mean_err'] > high_thresh * 0.6 else '#7ee787'
        ax_main.set_title(
            f'Frame {fr["idx"]}  |  mean_err={fr["mean_err"]:.4f}',
            color=badge_col, fontsize=8, pad=2)

        # ── Bottom-left: depth ────────────────────────────────────────────────
        ax_dep = fig.add_subplot(inner[1, 0])
        ax_dep.imshow(depth, cmap='inferno', aspect='auto')
        ax_dep.set_title('Depth', color='white', fontsize=7, pad=1)
        ax_dep.axis('off')

        # ── Bottom-right: pure error map ──────────────────────────────────────
        ax_err = fig.add_subplot(inner[1, 1])
        ax_err.imshow(err_hw, cmap='jet', aspect='auto')
        ax_err.set_title('Error', color='white', fontsize=7, pad=1)
        ax_err.axis('off')

    # Hide unused cells
    for flat_i in range(len(frame_results), n_groups * COLS):
        row_g = flat_i // COLS
        col_i = flat_i % COLS
        try:
            inner = outer[row_g, col_i].subgridspec(2, 2)
            for r in range(2):
                for c in range(2):
                    fig.add_subplot(inner[r, c]).axis('off')
        except Exception:
            pass

    out = os.path.join(SAVE_DIR, f'spatial_error_seq{seq}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    saved_files.append(out)
    print(f"  Saved: {out}")

    # ── Error statistics: which frame regions fail most? ──────────────────────
    # Aggregate error maps: average across all sampled frames to find persistent fail zones
    err_stack = np.stack([fr['err_hw'] for fr in frame_results], axis=0)
    mean_spatial = err_stack.mean(0)     # (H, W) — where does the model always fail?
    std_spatial  = err_stack.std(0)      # (H, W) — where is it inconsistent?

    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 5))
    fig2.patch.set_facecolor('#0d1117')

    panels = [
        (mean_spatial, 'hot',  'Mean Error Across All Frames\n(persistent fail zones)'),
        (std_spatial,  'cool', 'Std Dev of Error\n(inconsistent regions)'),
        (mean_spatial / (std_spatial + 1e-6), 'RdYlGn_r', 'Signal-to-Noise Ratio\n(high=systematically bad)'),
    ]
    for ax, (d, cm, title) in zip(axes2, panels):
        im = ax.imshow(d, cmap=cm, aspect='auto')
        plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.03)
        ax.set_title(title, color='white', fontsize=10, fontweight='bold')
        ax.axis('off')

        # Draw image region labels (sky, road, vehicles, sides)
        h, w = d.shape
        regions = [
            (0,       h//3,   'Sky / Horizon'),
            (h//3,    2*h//3, 'Mid (vehicles)'),
            (2*h//3,  h,      'Road / Ground'),
        ]
        for y0, y1, label in regions:
            ax.axhline(y0, color='white', linewidth=0.5, alpha=0.4)
            ax.text(w*0.01, (y0+y1)/2, label,
                    color='white', fontsize=8, va='center', alpha=0.7)

    fig2.suptitle(f'Seq {seq} — Aggregate Spatial Error (over {len(frame_results)} sampled frames)',
                  color='white', fontsize=12, fontweight='bold')
    out2 = os.path.join(SAVE_DIR, f'aggregate_error_zones_seq{seq}.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig2)
    saved_files.append(out2)
    print(f"  Saved: {out2}")

# ── Zip and download ──────────────────────────────────────────────────────────
zip_path = '/content/sgvo_spatial_errors.zip'
with ZipFile(zip_path, 'w') as zf:
    for f in saved_files:
        zf.write(f, os.path.basename(f))
        print(f"  + {os.path.basename(f)}")

print(f"\nDownloading ({os.path.getsize(zip_path)/1e6:.1f} MB)...")
files.download(zip_path)
