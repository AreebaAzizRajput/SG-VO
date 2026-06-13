
# ============================================================
# SG-VO ONLINE EVALUATION VISUALIZER
# Paste this entire cell into Colab and run it.
# It runs a mini online-eval loop (first N frames) and
# shows: input images, depth maps, losses, and trajectory.
# ============================================================

import sys, os, traceback, copy, time
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from imageio import imread
from skimage.transform import resize as imresize
from skimage import img_as_ubyte
from path import Path
from IPython.display import clear_output

sys.path.insert(0, '/content/SG-VO')

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEQUENCE       = '09'
DATASET_DIR    = '/content/SG-VO/data/kitti_odom/sequences/'
POSE_NET_PATH  = '/content/SG-VO/checkpoints/exp_pose112_model_best.pth.tar'
DISP_NET_PATH  = '/content/SG-VO/checkpoints/dispnet112_model_best.pth.tar'
IMG_H, IMG_W   = 256, 832
VIZ_EVERY      = 10       # Show visualization every N frames
N_FRAMES       = 50       # How many frames to run (None = full sequence)
EPOCHS         = 2
LR             = 1e-4
PHOTO_W        = 1.0
SMOOTH_W       = 0.1
GEO_W          = 0.5
RESNET_LAYERS  = 50
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Running on: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Step 1: Imports ───────────────────────────────────────────────────────────
print("\n[1/6] Importing SG-VO modules...")
try:
    from models import DispResNet, PoseResNet
    from utils import compute_depth, compute_pose_with_inv, extract_orb, tensor2array
    from loss_functions import compute_photo_and_geometry_loss, compute_smooth_loss
    from inverse_warp import pose_vec2mat
    print("     OK")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
    raise

# ── Step 2: Load models ───────────────────────────────────────────────────────
print("\n[2/6] Loading pretrained models...")
try:
    weights_pose = torch.load(POSE_NET_PATH, map_location=device)
    pose_net = PoseResNet().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=True)
    print(f"     PoseNet loaded from {POSE_NET_PATH}")

    weights_disp = torch.load(DISP_NET_PATH, map_location=device)
    disp_net = DispResNet(RESNET_LAYERS, False).to(device)
    disp_net.load_state_dict(weights_disp['state_dict'], strict=True)
    print(f"     DispNet loaded from {DISP_NET_PATH}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
    raise

# ── Step 3: Load dataset ──────────────────────────────────────────────────────
print("\n[3/6] Loading dataset...")
try:
    image_dir = Path(DATASET_DIR + SEQUENCE + '/image_2/')
    test_files = sorted(image_dir.files('*.png') + image_dir.files('*.jpg'))
    n_total = len(test_files)
    n = min(N_FRAMES if N_FRAMES else n_total, n_total)
    print(f"     Sequence {SEQUENCE}: {n_total} images total, running first {n}")

    cam_txt = image_dir / 'cam.txt'
    if not cam_txt.exists():
        raise FileNotFoundError(f"cam.txt not found at {cam_txt}")
    K = np.genfromtxt(cam_txt).astype(np.float32).reshape(3, 3)
    intrinsics = torch.from_numpy(K).unsqueeze(0).to(device)
    print(f"     Intrinsics loaded: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
    raise

# ── Helper functions ──────────────────────────────────────────────────────────
def load_tensor(path):
    img = imread(path)
    if img.shape[0] != IMG_H or img.shape[1] != IMG_W:
        img = img_as_ubyte(imresize(img, (IMG_H, IMG_W)))
    img = extract_orb(img).astype(np.float32)
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0) / 255.0
    t[:, :3] = (t[:, :3] - 0.45) / 0.225
    return t

def update_pose(global_pose, pose_vec):
    mat = pose_vec2mat(pose_vec).squeeze(0).cpu().detach().numpy()
    mat = np.vstack([mat, [0, 0, 0, 1]])
    return global_pose @ np.linalg.inv(mat)

def make_figure():
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
    return fig, gs

def colorize_depth(disp_tensor):
    d = disp_tensor.squeeze().cpu().detach().numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return plt.cm.inferno(d)[:, :, :3]

def denorm_img(tensor):
    t = tensor.squeeze()[:3].cpu().detach().numpy().transpose(1, 2, 0)
    t = t * 0.225 + 0.45
    return np.clip(t, 0, 1)

# ── Step 4: Init tracking ─────────────────────────────────────────────────────
print("\n[4/6] Initialising online adaptation loop...")
optimizer = torch.optim.Adam(
    [{'params': disp_net.parameters(), 'lr': LR},
     {'params': pose_net.parameters(), 'lr': LR}],
    betas=(0.9, 0.999), weight_decay=0
)
weights_disp_best = copy.deepcopy(disp_net.state_dict())
weights_pose_best = copy.deepcopy(pose_net.state_dict())

global_pose = np.eye(4)
trajectory  = [global_pose[:3, :].reshape(1, 12)]

loss_history = {'total': [], 'photo': [], 'smooth': [], 'geo': []}
frame_times  = []

print("     OK — starting loop\n")

# ── Step 5: Main loop ─────────────────────────────────────────────────────────
print("[5/6] Running online adaptation...")

for i in range(n - 1):
    t0 = time.time()

    try:
        tgt = load_tensor(test_files[i]).to(device)
        ref = load_tensor(test_files[i + 1]).to(device)
    except Exception as e:
        print(f"\nFAILED loading frame {i}: {e}")
        traceback.print_exc()
        break

    best_error = -1.0

    # ── Online adaptation epochs ──────────────────────────────────────────────
    try:
        for ep in range(EPOCHS):
            disp_net.train(); pose_net.train()
            tgt_d = tgt[:, :3]
            ref_d = ref[:, :3]
            tgt_depth, ref_depths = compute_depth(disp_net, tgt_d, [ref_d])
            poses, poses_inv = compute_pose_with_inv(pose_net, tgt, [ref])
            l1, l3 = compute_photo_and_geometry_loss(
                tgt_d, [ref_d], intrinsics,
                tgt_depth, ref_depths, poses, poses_inv,
                1, 1, 1, 0, 'zeros')
            l2 = compute_smooth_loss(tgt_depth, tgt_d, ref_depths, [ref_d])
            loss = PHOTO_W * l1 + SMOOTH_W * l2 + GEO_W * l3

            is_best = (best_error < 0) or (l1.item() < best_error)
            if is_best:
                best_error = l1.item()
                weights_disp_best = copy.deepcopy(disp_net.state_dict())
                weights_pose_best = copy.deepcopy(pose_net.state_dict())

            if ep < EPOCHS - 1:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        loss_history['total'].append(loss.item())
        loss_history['photo'].append(l1.item())
        loss_history['smooth'].append(l2.item())
        loss_history['geo'].append(l3.item())

    except Exception as e:
        print(f"\nFAILED during adaptation at frame {i}: {e}")
        traceback.print_exc()
        break

    # ── Load best weights and infer pose ─────────────────────────────────────
    try:
        pose_net.load_state_dict(weights_pose_best, strict=True)
        disp_net.load_state_dict(weights_disp_best, strict=True)
        optimizer = torch.optim.Adam(
            [{'params': disp_net.parameters(), 'lr': LR},
             {'params': pose_net.parameters(), 'lr': LR}],
            betas=(0.9, 0.999), weight_decay=0)

        disp_net.eval(); pose_net.eval()
        with torch.no_grad():
            pose_vec  = pose_net(tgt, ref)
            disp_pred = disp_net(tgt[:, :3])[0]

        global_pose = update_pose(global_pose, pose_vec)
        trajectory.append(global_pose[:3, :].reshape(1, 12))

    except Exception as e:
        print(f"\nFAILED during pose inference at frame {i}: {e}")
        traceback.print_exc()
        break

    frame_times.append(time.time() - t0)

    # ── Visualisation ─────────────────────────────────────────────────────────
    if (i % VIZ_EVERY == 0) or (i == n - 2):
        clear_output(wait=True)
        fig, gs = make_figure()
        txt_kw = dict(color='white', fontsize=10, fontweight='bold')

        # Row 0: images
        ax_tgt = fig.add_subplot(gs[0, 0])
        ax_tgt.imshow(denorm_img(tgt)); ax_tgt.axis('off')
        ax_tgt.set_title(f'Target (frame {i})', **txt_kw)

        ax_ref = fig.add_subplot(gs[0, 1])
        ax_ref.imshow(denorm_img(ref)); ax_ref.axis('off')
        ax_ref.set_title(f'Reference (frame {i+1})', **txt_kw)

        # Depth map
        ax_dep = fig.add_subplot(gs[0, 2:])
        ax_dep.imshow(colorize_depth(disp_pred)); ax_dep.axis('off')
        ax_dep.set_title('Predicted Depth (frame {})'.format(i), **txt_kw)

        # Row 1: Loss curves
        lax = fig.add_subplot(gs[1, :2])
        lax.set_facecolor('#161b22')
        colors = ['#58a6ff', '#f78166', '#7ee787', '#ffa657']
        for key, col, lbl in zip(
                ['total', 'photo', 'smooth', 'geo'], colors,
                ['Total', 'Photo', 'Smooth', 'Geo Consistency']):
            lax.plot(loss_history[key], color=col, linewidth=1.5, label=lbl)
        lax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
        lax.set_title('Loss History', **txt_kw)
        lax.tick_params(colors='white'); lax.set_facecolor('#161b22')
        for sp in lax.spines.values(): sp.set_color('#30363d')

        # Frame timing
        tax = fig.add_subplot(gs[1, 2:])
        tax.set_facecolor('#161b22')
        tax.plot(frame_times, color='#d2a8ff', linewidth=1.5)
        tax.set_title(f'Frame time (avg {np.mean(frame_times):.2f}s)', **txt_kw)
        tax.tick_params(colors='white')
        for sp in tax.spines.values(): sp.set_color('#30363d')

        # Row 2: Trajectory (X-Z top-down)
        traj_np = np.concatenate(trajectory, axis=0)  # (N, 12)
        xs = traj_np[:, 3]   # tx
        zs = traj_np[:, 11]  # tz
        trax = fig.add_subplot(gs[2, :])
        trax.set_facecolor('#161b22')
        sc = trax.scatter(xs, zs, c=np.arange(len(xs)),
                          cmap='plasma', s=6, zorder=3)
        trax.plot(xs, zs, color='#58a6ff', linewidth=0.8, alpha=0.5)
        trax.scatter([xs[0]], [zs[0]], c='lime', s=80, zorder=5, label='Start')
        trax.scatter([xs[-1]], [zs[-1]], c='red', s=80, zorder=5, label='Current')
        cbar = plt.colorbar(sc, ax=trax, orientation='vertical', pad=0.01)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.set_label('Frame', color='white', fontsize=8)
        trax.set_title(f'Trajectory — Seq {SEQUENCE} (X-Z top-down, frame {i}/{n})', **txt_kw)
        trax.set_xlabel('X (m)', color='white'); trax.set_ylabel('Z (m)', color='white')
        trax.tick_params(colors='white'); trax.legend(facecolor='#21262d', labelcolor='white')
        trax.set_aspect('equal'); trax.grid(True, color='#30363d', linewidth=0.5)
        for sp in trax.spines.values(): sp.set_color('#30363d')

        fig.suptitle(
            f'SG-VO Online Adaptation | Seq {SEQUENCE} | Frame {i}/{n} | '
            f'Loss: {loss.item():.4f} | Photo: {l1.item():.4f}',
            color='white', fontsize=13, fontweight='bold', y=1.01)
        plt.show()
        print(f"Frame {i:4d}/{n} | total={loss.item():.4f} "
              f"photo={l1.item():.4f} smooth={l2.item():.4f} "
              f"geo={l3.item():.4f} | {frame_times[-1]:.2f}s/frame")

# ── Step 6: Save trajectory ───────────────────────────────────────────────────
print("\n[6/6] Saving trajectory...")
traj_np = np.concatenate(trajectory, axis=0)
out_path = f'/content/SG-VO/vo_results_online/{SEQUENCE}_viz.txt'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savetxt(out_path, traj_np, delimiter=' ', fmt='%1.8e')
print(f"     Saved {traj_np.shape[0]} poses to {out_path}")
print("\nDone!")
