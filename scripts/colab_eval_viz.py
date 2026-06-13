"""
SG-VO Paper Claims Verifier
Visualizes trajectory errors vs ground truth to verify paper's reported metrics.
Paste: exec(open('/content/SG-VO/scripts/colab_eval_viz.py').read())
"""

import os, sys, subprocess, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from zipfile import ZipFile
from google.colab import files

sys.path.insert(0, '/content/SG-VO')

SAVE_DIR   = '/content/SG-VO/viz_export'
GT_DIR     = '/content/SG-VO/kitti_eval/gt_poses'
PRED_DIRS  = {
    'offline': '/content/SG-VO/vo_results',
    'online':  '/content/SG-VO/vo_results_online',
}
SEQUENCES  = ['09', '10']
os.makedirs(SAVE_DIR, exist_ok=True)

# Paper's reported numbers (Table 1 in SG-VO paper) for reference lines
PAPER_CLAIMS = {
    'offline': {
        '09': {'t_err': 7.08, 'r_err': 2.48},
        '10': {'t_err': 8.72, 'r_err': 3.11},
    },
    'online': {
        '09': {'t_err': 5.21, 'r_err': 1.93},
        '10': {'t_err': 6.74, 'r_err': 2.57},
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_poses(fpath):
    """Load Nx12 pose file and return list of 4x4 matrices."""
    data = np.loadtxt(fpath)
    poses = []
    for row in data:
        mat = np.eye(4)
        mat[:3, :] = row.reshape(3, 4)
        poses.append(mat)
    return poses

def poses_to_xyz(poses):
    return np.array([p[:3, 3] for p in poses])

def rotation_error(pose_err):
    """Rotation error in degrees from a 4x4 error matrix."""
    a = pose_err[0, 0]
    b = pose_err[1, 1]
    c = pose_err[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    d = np.clip(d, -1.0, 1.0)
    return np.degrees(np.arccos(d))

def translation_error(pose_err):
    return np.linalg.norm(pose_err[:3, 3])

def compute_segment_errors(gt_poses, pred_poses, lengths=(100, 200, 300, 400, 500, 600, 700, 800)):
    """Compute t_err and r_err over fixed-length subsequences (KITTI protocol)."""
    n = len(gt_poses)
    # Build cumulative distance along GT path
    cum_dist = [0.0]
    for i in range(1, n):
        diff = gt_poses[i][:3, 3] - gt_poses[i-1][:3, 3]
        cum_dist.append(cum_dist[-1] + np.linalg.norm(diff))

    errors = []
    for start in range(n):
        for length in lengths:
            # Find end index where cumulative distance >= start_dist + length
            start_dist = cum_dist[start]
            end = start
            for j in range(start, n):
                if cum_dist[j] - start_dist >= length:
                    end = j
                    break
            else:
                continue
            if end == start:
                continue

            # Relative pose in GT and predicted
            gt_rel   = np.linalg.inv(gt_poses[start]) @ gt_poses[end]
            pred_rel = np.linalg.inv(pred_poses[start]) @ pred_poses[end]
            pose_err = np.linalg.inv(gt_rel) @ pred_rel

            t_err = translation_error(pose_err) / length   # as fraction of length
            r_err = rotation_error(pose_err) / length       # deg/m

            errors.append({
                'start': start,
                'end':   end,
                'length': length,
                't_err': t_err * 100,   # percent
                'r_err': r_err * 100,   # deg/100m
            })
    return errors

def compute_ate(gt_poses, pred_poses):
    """Absolute Trajectory Error after scale + rigid alignment."""
    gt_xyz   = poses_to_xyz(gt_poses)
    pred_xyz = poses_to_xyz(pred_poses)
    n = min(len(gt_xyz), len(pred_xyz))
    gt_xyz = gt_xyz[:n]; pred_xyz = pred_xyz[:n]

    # Centre
    gt_c   = gt_xyz.mean(0)
    pred_c = pred_xyz.mean(0)
    gt_z   = gt_xyz - gt_c
    pred_z = pred_xyz - pred_c

    # Optimal scale
    scale = np.sqrt((gt_z**2).sum() / (pred_z**2).sum() + 1e-9)
    pred_z_s = pred_z * scale

    # Optimal rotation (SVD)
    H  = pred_z_s.T @ gt_z
    U, _, Vt = np.linalg.svd(H)
    R  = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    aligned = (R @ pred_z_s.T).T + gt_c
    diff    = gt_xyz - aligned
    return np.sqrt((diff**2).sum(1)), aligned, scale

# ── Main loop: evaluate each mode × sequence ─────────────────────────────────

all_results = {}   # mode -> seq -> dict

for mode, pred_dir in PRED_DIRS.items():
    all_results[mode] = {}
    for seq in SEQUENCES:
        # Find pred file (may be named 09.txt or 09_quickviz.txt etc.)
        pred_file = None
        for cand in [f'{seq}.txt', f'{seq}_quickviz.txt', f'{seq}_viz.txt']:
            p = os.path.join(pred_dir, cand)
            if os.path.exists(p):
                pred_file = p; break

        gt_file = os.path.join(GT_DIR, f'{seq}.txt')

        if pred_file is None:
            print(f"[{mode}] seq {seq}: no prediction file found in {pred_dir}")
            continue
        if not os.path.exists(gt_file):
            print(f"[{mode}] seq {seq}: GT file missing at {gt_file}")
            continue

        gt_poses   = load_poses(gt_file)
        pred_poses = load_poses(pred_file)
        n = min(len(gt_poses), len(pred_poses))
        gt_poses   = gt_poses[:n]
        pred_poses = pred_poses[:n]

        # Segment errors
        seg_errors = compute_segment_errors(gt_poses, pred_poses)
        if seg_errors:
            t_errs = [e['t_err'] for e in seg_errors]
            r_errs = [e['r_err'] for e in seg_errors]
            mean_t = np.mean(t_errs)
            mean_r = np.mean(r_errs)
        else:
            mean_t = mean_r = 0.0

        # ATE
        ate_per_frame, pred_aligned, scale = compute_ate(gt_poses, pred_poses)
        mean_ate = float(ate_per_frame.mean())

        gt_xyz   = poses_to_xyz(gt_poses)
        pred_xyz = poses_to_xyz(pred_poses)

        all_results[mode][seq] = {
            'gt_poses':      gt_poses,
            'pred_poses':    pred_poses,
            'gt_xyz':        gt_xyz,
            'pred_aligned':  pred_aligned,
            'ate_per_frame': ate_per_frame,
            'mean_ate':      mean_ate,
            'mean_t':        mean_t,
            'mean_r':        mean_r,
            'seg_errors':    seg_errors,
            'scale':         scale,
            'n':             n,
        }
        paper = PAPER_CLAIMS.get(mode, {}).get(seq, {})
        print(f"[{mode}] Seq {seq}: t_err={mean_t:.2f}% (paper: {paper.get('t_err','?')}%)  "
              f"r_err={mean_r:.2f}°/100m (paper: {paper.get('r_err','?')})  "
              f"ATE={mean_ate:.3f}m  scale={scale:.3f}")

# ── Figures ───────────────────────────────────────────────────────────────────

DARK_BG  = '#0d1117'
PANEL_BG = '#161b22'
GRID_COL = '#30363d'
C_GT     = '#7ee787'    # green  — ground truth
C_OFF    = '#58a6ff'    # blue   — offline pred
C_ONL    = '#f78166'    # orange — online pred
C_ATE    = '#d2a8ff'    # purple — ATE
C_CLAIM  = '#ffa657'    # yellow — paper claim line

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color='white', fontsize=10, fontweight='bold')
    ax.set_xlabel(xlabel, color='white', fontsize=9)
    ax.set_ylabel(ylabel, color='white', fontsize=9)
    ax.tick_params(colors='white', labelsize=8)
    ax.grid(True, color=GRID_COL, linewidth=0.5)
    for sp in ax.spines.values(): sp.set_color(GRID_COL)

saved_files = []

for seq in SEQUENCES:
    # Check we have at least one mode
    modes_avail = [m for m in PRED_DIRS if seq in all_results.get(m, {})]
    if not modes_avail:
        print(f"Seq {seq}: no results available, skipping plot.")
        continue

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

    # ── Row 0: Trajectory overlays ───────────────────────────────────────────
    ax_traj = fig.add_subplot(gs[0, :2])
    style_ax(ax_traj, title=f'Seq {seq} — Predicted vs GT Trajectory (scale-aligned)',
             xlabel='X (m)', ylabel='Z (m)')

    gt_xyz = all_results[modes_avail[0]][seq]['gt_xyz']
    ax_traj.plot(gt_xyz[:, 0], gt_xyz[:, 2], color=C_GT,
                 linewidth=2.0, label='Ground Truth', zorder=5)

    for mode, col in [('offline', C_OFF), ('online', C_ONL)]:
        if seq not in all_results.get(mode, {}):
            continue
        r = all_results[mode][seq]
        ax_traj.plot(r['pred_aligned'][:, 0], r['pred_aligned'][:, 2],
                     color=col, linewidth=1.2, linestyle='--',
                     label=f'{mode.capitalize()} (t={r["mean_t"]:.1f}%)', zorder=4)

    ax_traj.scatter([gt_xyz[0, 0]], [gt_xyz[0, 2]], c='lime', s=100, zorder=6)
    ax_traj.scatter([gt_xyz[-1, 0]], [gt_xyz[-1, 2]], c='red', s=100, zorder=6)
    ax_traj.set_aspect('equal')
    ax_traj.legend(facecolor='#21262d', labelcolor='white', fontsize=9)

    # ── Row 0 col 2: Summary stats box ───────────────────────────────────────
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.set_facecolor(PANEL_BG); ax_stats.axis('off')
    lines = [f'Sequence {seq} — Summary\n']
    for mode in ['offline', 'online']:
        if seq not in all_results.get(mode, {}):
            continue
        r  = all_results[mode][seq]
        pc = PAPER_CLAIMS.get(mode, {}).get(seq, {})
        lines.append(f'── {mode.upper()} ──')
        lines.append(f't_err : {r["mean_t"]:.2f}%  (paper: {pc.get("t_err","?")}%)')
        lines.append(f'r_err : {r["mean_r"]:.2f}°/100m  (paper: {pc.get("r_err","?")})')
        lines.append(f'ATE   : {r["mean_ate"]:.3f} m')
        lines.append(f'Frames: {r["n"]}')
        lines.append(f'Scale : {r["scale"]:.4f}')
        lines.append('')
    ax_stats.text(0.05, 0.95, '\n'.join(lines), transform=ax_stats.transAxes,
                  color='white', fontsize=9, va='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', fc='#21262d', ec='#30363d'))
    ax_stats.set_title('Metrics vs Paper Claims', color='white', fontsize=10, fontweight='bold')

    # ── Row 1: ATE over time ─────────────────────────────────────────────────
    ax_ate = fig.add_subplot(gs[1, :])
    style_ax(ax_ate, title=f'Seq {seq} — ATE per Frame (lower=better)',
             xlabel='Frame index', ylabel='ATE (m)')
    for mode, col in [('offline', C_OFF), ('online', C_ONL)]:
        if seq not in all_results.get(mode, {}):
            continue
        r = all_results[mode][seq]
        frames = np.arange(r['n'])
        ax_ate.plot(frames, r['ate_per_frame'], color=col, linewidth=0.9,
                    alpha=0.85, label=f'{mode.capitalize()} (mean={r["mean_ate"]:.3f}m)')
        ax_ate.axhline(r['mean_ate'], color=col, linestyle=':', linewidth=1.5, alpha=0.6)
    ax_ate.legend(facecolor='#21262d', labelcolor='white', fontsize=9)

    # ── Row 2 left: t_err by segment length ──────────────────────────────────
    ax_terr = fig.add_subplot(gs[2, :2])
    style_ax(ax_terr, title=f'Seq {seq} — Translation Error by Segment Length',
             xlabel='Segment length (m)', ylabel='t_err (%)')

    for mode, col in [('offline', C_OFF), ('online', C_ONL)]:
        if seq not in all_results.get(mode, {}):
            continue
        r   = all_results[mode][seq]
        seg = r['seg_errors']
        if not seg:
            continue
        lengths   = sorted(set(e['length'] for e in seg))
        mean_by_l = [np.mean([e['t_err'] for e in seg if e['length'] == l])
                     for l in lengths]
        ax_terr.plot(lengths, mean_by_l, color=col, marker='o', linewidth=1.5,
                     markersize=5, label=f'{mode.capitalize()} measured')
        pc = PAPER_CLAIMS.get(mode, {}).get(seq, {})
        if 't_err' in pc:
            ax_terr.axhline(pc['t_err'], color=col, linestyle='--', linewidth=1.0,
                            alpha=0.6, label=f'{mode.capitalize()} paper claim ({pc["t_err"]}%)')

    ax_terr.legend(facecolor='#21262d', labelcolor='white', fontsize=9)

    # ── Row 2 right: r_err by segment length ─────────────────────────────────
    ax_rerr = fig.add_subplot(gs[2, 2])
    style_ax(ax_rerr, title=f'Seq {seq} — Rotation Error by Segment',
             xlabel='Segment length (m)', ylabel='r_err (°/100m)')

    for mode, col in [('offline', C_OFF), ('online', C_ONL)]:
        if seq not in all_results.get(mode, {}):
            continue
        r   = all_results[mode][seq]
        seg = r['seg_errors']
        if not seg:
            continue
        lengths   = sorted(set(e['length'] for e in seg))
        mean_by_l = [np.mean([e['r_err'] for e in seg if e['length'] == l])
                     for l in lengths]
        ax_rerr.plot(lengths, mean_by_l, color=col, marker='s', linewidth=1.5,
                     markersize=5, label=f'{mode.capitalize()} measured')
        pc = PAPER_CLAIMS.get(mode, {}).get(seq, {})
        if 'r_err' in pc:
            ax_rerr.axhline(pc['r_err'], color=col, linestyle='--', linewidth=1.0,
                            alpha=0.6, label=f'{mode.capitalize()} paper ({pc["r_err"]})')

    ax_rerr.legend(facecolor='#21262d', labelcolor='white', fontsize=9)

    fig.suptitle(f'SG-VO Error Analysis — Sequence {seq}  |  '
                 f'Green = GT  |  Blue = Offline  |  Orange = Online',
                 color='white', fontsize=13, fontweight='bold')

    out = os.path.join(SAVE_DIR, f'error_analysis_seq{seq}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    saved_files.append(out)
    print(f"Saved: {out}")

# ── Combined comparison bar chart ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(DARK_BG)

metrics   = ['t_err (%)', 'r_err (°/100m)']
keys      = ['mean_t', 'mean_r']
paper_keys = ['t_err', 'r_err']

for ax, metric, key, pk in zip(axes, metrics, keys, paper_keys):
    ax.set_facecolor(PANEL_BG)
    labels, measured_vals, paper_vals, bar_cols = [], [], [], []

    for seq in SEQUENCES:
        for mode, col in [('offline', C_OFF), ('online', C_ONL)]:
            if seq not in all_results.get(mode, {}):
                continue
            r = all_results[mode][seq]
            labels.append(f'Seq{seq}\n{mode[:3].title()}')
            measured_vals.append(r[key])
            paper_vals.append(PAPER_CLAIMS.get(mode, {}).get(seq, {}).get(pk, 0))
            bar_cols.append(col)

    x = np.arange(len(labels))
    w = 0.35
    bars_m = ax.bar(x - w/2, measured_vals, w, label='Measured', color=bar_cols, alpha=0.85)
    bars_p = ax.bar(x + w/2, paper_vals,    w, label='Paper claim',
                    color=[c + '55' for c in bar_cols], edgecolor=bar_cols, linewidth=1.5)

    ax.set_xticks(x); ax.set_xticklabels(labels, color='white', fontsize=9)
    ax.set_title(metric, color='white', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric, color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#21262d', labelcolor='white')
    ax.grid(True, axis='y', color=GRID_COL, linewidth=0.5)
    for sp in ax.spines.values(): sp.set_color(GRID_COL)

    # Value labels on bars
    for bar in bars_m:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f'{h:.2f}', ha='center', color='white', fontsize=8)
    for bar in bars_p:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f'{h:.2f}', ha='center', color='#ffa657', fontsize=8)

fig.suptitle('SG-VO: Measured vs Paper-Claimed Errors  |  Solid=Measured  Transparent=Paper',
             color='white', fontsize=12, fontweight='bold')
out = os.path.join(SAVE_DIR, 'paper_claim_comparison.png')
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close(fig)
saved_files.append(out)
print(f"Saved: {out}")

# ── Zip and download ──────────────────────────────────────────────────────────
zip_path = '/content/sgvo_error_analysis.zip'
with ZipFile(zip_path, 'w') as zf:
    for f in saved_files:
        zf.write(f, os.path.basename(f))
        print(f"  + {os.path.basename(f)}")

print(f"\nDownloading {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)...")
files.download(zip_path)
