import argparse
import time
import copy
import concurrent.futures

import torch
import numpy as np
from tqdm import tqdm
from imageio.v2 import imread, imsave
from skimage.transform import resize as imresize
from skimage import img_as_ubyte
from path import Path

from models import DispResNet, PoseResNet
from logger import AverageMeter
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_ssim_loss
from inverse_warp import inverse_warp2
from utils import (
    tensor2array,
    compute_depth,
    compute_pose_with_inv,
    extract_orb,
)
from inverse_warp import pose_vec2mat
from lora import (
    inject_lora_pose_net, freeze_all,
    lora_parameters, lora_state_dict, load_lora_state_dict, count_lora_params,
    set_lora_enabled, reset_lora,
)

parser = argparse.ArgumentParser(description='Inference with online adaptation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", required=True, type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="Do not resize if true")
parser.add_argument("--thread", action="store_true", help="Preprocess images with multi-threads")

parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

parser.add_argument("--sequence", default='09', type=str, help="sequence to test")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# online tuning arguments
parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--epochs', default=2, type=int, metavar='N', help='epochs trained for each snippet')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1, help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=0, help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int,  default=1, help='with or without imagenet pretrain for resnet')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for online training', default=2)

# for depth
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
# for partial adaptation (decoder only, original flag)
parser.add_argument("--part", action='store_true', help="partial adaptation: train decoder layers only")
# for best selection
parser.add_argument("--select-best", type=int, default=1, help="select best parameters or not")

# ── LoRA arguments ────────────────────────────────────────────────────────────
# What LoRA is: instead of updating all N parameters of a weight matrix during
# adaptation, we freeze the original matrix and learn a small "correction"
# ΔW = B×A where A has shape (rank × in) and B has shape (out × rank).
# rank << min(in, out), so we update far fewer numbers.
# Setting --lora-rank 0 disables LoRA and falls back to the original full adaptation.
parser.add_argument('--lora-rank', type=int, default=0,
                    help='LoRA adapter rank. 0 = disabled (original full adaptation). '
                         'Try 4, 8, or 16 as starting values.')
parser.add_argument('--lora-targets', type=str, default='attention',
                    choices=['attention', 'decoder', 'both'],
                    help='Which PoseNet modules receive LoRA adapters. '
                         '"attention" = Wq/Wk/Wv + projection convs in cross-attention (~25K params at rank 8). '
                         '"decoder" = four convs in PoseDecoder (~200K params at rank 8). '
                         '"both" = all of the above.')
parser.add_argument('--lora-alpha', type=float, default=1.0,
                    help='LoRA scaling factor. Effective lr scale = lora_alpha / lora_rank. '
                         'Increasing alpha amplifies the adapter contribution.')

# ── Trigger arguments ─────────────────────────────────────────────────────────
# What the trigger does: before every frame, do one cheap no-gradient forward
# pass to measure the current photometric loss.  If that loss is below
# (adapt_threshold × running_average_loss), the model is already doing well
# on this frame and we skip the expensive adaptation step entirely.
# 0.0 = always adapt (original behaviour).  Try 0.8 as a first experiment.
parser.add_argument('--probe-only', action='store_true',
                    help='Never adapt; just record the per-frame probe loss to '
                         'probe_losses_<seq>.txt for offline trigger design.')
parser.add_argument('--adapt-threshold', type=float, default=0.0,
                    help='LEGACY spike trigger (kept for ablation): adapt only when '
                         'photo_loss > adapt_threshold × EMA_loss. Fires at the '
                         'hardest frames, which measurably degrades accuracy. '
                         '0.0 = disabled. Prefer --cusum-h.')
parser.add_argument('--cusum-h', type=float, default=0.0,
                    help='CUSUM trigger decision threshold in units of reference '
                         'std (e.g. 16). Adapt when the CUSUM statistic S_t = '
                         'max(0, S + (loss - mu_ref - kappa)) exceeds h*sigma_ref: '
                         'fires only on SUSTAINED loss elevation (domain shift), '
                         'not transient hard frames. 0.0 = disabled.')
parser.add_argument('--cusum-kappa', type=float, default=1.0,
                    help='CUSUM allowance in units of reference std (default 1.0).')
parser.add_argument('--cusum-calib-frames', type=int, default=100,
                    help='Frames used to estimate (mu_ref, sigma_ref) at sequence '
                         'start when no explicit reference is given (no adaptation '
                         'during calibration).')
parser.add_argument('--cusum-ref-mean', type=float, default=None,
                    help='Explicit reference loss mean (e.g. training-domain '
                         'statistics). Overrides start-of-sequence calibration; '
                         'REQUIRED for sequences that begin already domain-shifted '
                         '(e.g. vKITTI fog).')
parser.add_argument('--cusum-ref-std', type=float, default=None,
                    help='Explicit reference loss std, together with --cusum-ref-mean.')
parser.add_argument('--probe-signal', type=str, default='photo',
                    choices=['photo', 'ratio', 'bnstats'],
                    help='Signal fed to the trigger/probe trace. photo: raw '
                         'photometric loss (legacy — confounded by image '
                         'contrast: fog LOWERS it while accuracy collapses). '
                         'ratio: photometric loss of the predicted pose divided '
                         'by that of the identity (zero-motion) warp — the '
                         'contrast factor cancels, so it measures how much '
                         'motion the pose actually explains. bnstats: mean '
                         'BatchNorm feature-statistics distance from the '
                         'training domain — absolute, non-photometric.')
parser.add_argument('--adapt-loss', type=str, default='photo',
                    choices=['photo', 'norm'],
                    help='Objective for the adaptation step. photo: raw '
                         'photometric loss (legacy). norm: photometric loss '
                         'divided by the zero-motion baseline of the frame — '
                         'low contrast (fog) shrinks raw-loss gradients by the '
                         'same transmission factor that confounds the probe, '
                         'making adaptation sluggish exactly when it is '
                         'needed; the division cancels that factor.')
parser.add_argument('--reset-on-recovery', action='store_true',
                    help='Undoable adaptation (LoRA + CUSUM only): while '
                         'shifted, also probe with adapters DISABLED (the '
                         'pristine base model — asks whether the WORLD is '
                         'still shifted, so an adapter that has fit the fog '
                         'cannot mask ongoing fog). When the base model '
                         'matches or beats the adapted model for '
                         '--recovery-patience consecutive frames — the '
                         'adapters have stopped helping — reset the adapters '
                         'to zero: the exact pre-adaptation model is '
                         'recovered by construction.')
parser.add_argument('--recovery-patience', type=int, default=30,
                    help='Consecutive frames the base model must match/beat '
                         'the adapted model before adapters are reset.')
parser.add_argument('--recovery-margin', type=float, default=0.1,
                    help='In units of the reference sigma: the adapter must '
                         'beat the base model by MORE than this margin to '
                         'count as still helping. Without it a dead-weight '
                         'adapter that wins by numerical dust (measured: '
                         '1e-5 on every frame) blocks the reset forever.')
parser.add_argument('--static-floor', type=float, default=0.3,
                    help='Static-frame gate for the ratio signal: when the '
                         'zero-motion baseline drops below this fraction of '
                         'its running median, the vehicle is (near-)still — '
                         'the ratio divides two near-zero losses and '
                         'explodes (measured up to 2.2 at a vKITTI stop, 16 '
                         'false fires). Such frames hold the trigger state '
                         'and skip adaptation: a static frame carries no '
                         'motion information to adapt on. 0 disables.')


def load_tensor_image(filename, args):
    img = imread(filename)
    h, w, _ = img.shape

    if not args.no_resize and (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width))
        img = img_as_ubyte(img)

    img = extract_orb(img).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))

    tensor = torch.from_numpy(img).unsqueeze(0) / 255.0
    tensor[:, :3] = (tensor[:, :3] - 0.45) / 0.225
    return tensor


def preload_images(test_files, args):
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(load_tensor_image, file, args) for file in test_files]
        print("loading tensors ...")
        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
        tensor_imgs_total = [future.result() for future in futures]
    return tensor_imgs_total


def init_models(args, device):
    weights_pose = torch.load(args.pretrained_posenet)
    pose_net = PoseResNet().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=True)

    weights_disp = torch.load(args.pretrained_disp)
    disp_net = DispResNet(args.resnet_layers, False).to(device)
    disp_net.load_state_dict(weights_disp['state_dict'], strict=True)

    return pose_net, disp_net, weights_pose, weights_disp


def build_optimizer(args, pose_net, disp_net, use_lora=False):
    if use_lora:
        # Only optimise LoRA adapter params in pose_net.
        # disp_net is fully frozen when using LoRA, so we exclude it.
        params = list(lora_parameters(pose_net))
        return torch.optim.Adam(params, lr=args.lr,
                                betas=(args.momentum, args.beta),
                                weight_decay=args.weight_decay)
    # Original: full adaptation of both networks
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr},
    ]
    return torch.optim.Adam(optim_params,
                            betas=(args.momentum, args.beta),
                            weight_decay=args.weight_decay)


def freeze_except_decoder(net):
    for name, param in net.named_parameters():
        param.requires_grad = ("decoder" in name)


def update_global_pose(global_pose, pose):
    pose_mat = pose_vec2mat(pose).squeeze(0).cpu().detach().numpy()
    pose_mat = np.vstack([pose_mat, [0, 0, 0, 1]])
    return global_pose @ np.linalg.inv(pose_mat)


def process_tail_poses(pose_net, test_files, start_idx,
                       seq_len, global_pose, poses, device, args):
    img1 = load_tensor_image(test_files[start_idx], args).to(device)

    for k in range(seq_len - 2):
        img2 = load_tensor_image(test_files[start_idx + k + 1], args).to(device)
        pose = pose_net(img1, img2)
        global_pose = update_global_pose(global_pose, pose)
        poses.append(global_pose[:3].reshape(1, 12))
        img1 = img2

    return global_pose, poses


def train(args, tgt_img, ref_imgs, intrinsics, disp_net, pose_net, optimizer, epochs,
          weights_disp_best, weights_pose_best, use_lora=False, photo_norm=None):
    global device
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter(i=4, precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    disp_net.train()
    pose_net.train()

    if use_lora:
        # Confine adaptation strictly to the adapters. BatchNorm running
        # stats are adapted state too: in train mode every adaptation step
        # drifts them toward the current (e.g. fog) batch statistics — state
        # that lives OUTSIDE the adapters, silently breaking reset_lora's
        # exact-recovery guarantee (measured as a +0.03 probe elevation after
        # reset) and the "only 0.083% of parameters adapt" accounting.
        # Pin ONLY the BatchNorm modules: whole-net eval() would also switch
        # DispResNet's forward to its eval shape (single tensor, not the
        # multi-scale list) and crash the loss (see 8576494).
        for net in (disp_net, pose_net):
            for m in net.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()

    end        = time.time()
    best_error = -1

    for i in range(epochs):
        data_time.update(time.time() - end)
        tgt_img    = tgt_img.to(device)
        ref_imgs   = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        tgt_img_forDepth  = tgt_img[:, :3, :, :]
        ref_imgs_forDepth = [img[:, :3, :, :] for img in ref_imgs]

        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img_forDepth, ref_imgs_forDepth)
        poses, poses_inv      = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(
            tgt_img_forDepth, ref_imgs_forDepth, intrinsics,
            tgt_depth, ref_depths, poses, poses_inv,
            args.num_scales, args.with_ssim,
            args.with_mask, args.with_auto_mask, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img_forDepth, ref_depths, ref_imgs_forDepth)

        if photo_norm is not None:
            # --adapt-loss norm: fog scales the photometric term (and its
            # gradients) down by the transmission factor; the zero-motion
            # baseline carries the same factor, so this division restores
            # contrast-invariant step sizes. The smooth/geometry terms are
            # not contrast-scaled and keep their original weighting.
            loss_1 = loss_1 / (photo_norm + 1e-12)

        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3
        losses.update([loss, loss_1, loss_2, loss_3])

        decisive_error = losses.val[1]  # photometric loss as selection metric

        if best_error < 0:
            best_error = decisive_error

        is_best    = decisive_error < best_error
        best_error = min(best_error, decisive_error)

        if is_best:
            if use_lora:
                # LoRA: snapshot is tens of KB instead of ~150 MB
                weights_pose_best = lora_state_dict(pose_net)
                # disp_net is fully frozen when using LoRA — its snapshot never changes
            else:
                weights_disp_best = copy.deepcopy(disp_net.state_dict())
                weights_pose_best = copy.deepcopy(pose_net.state_dict())

        if i < epochs - 1 or epochs == 1 or args.select_best == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return (losses.avg,
            ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss'],
            weights_disp_best,
            weights_pose_best)


def probe_photo_loss(args, tgt_img, ref_imgs, intrinsics, disp_net, pose_net) -> float:
    """Single no-gradient forward pass returning the photometric loss value.

    Used by the trigger to decide whether to adapt on the current frame.
    Cost: one forward pass (no backward), much cheaper than a training step.
    """
    disp_net.eval()
    pose_net.eval()

    def _depth_scales(disp_out):
        # DispResNet returns a list of scales in train mode but a single
        # tensor in eval mode (models/DispResNet.py forward). Normalise to
        # a list so we never iterate a tensor's batch dimension.
        if not isinstance(disp_out, (list, tuple)):
            disp_out = [disp_out]
        return [1 / d for d in disp_out]

    with torch.no_grad():
        tgt_d  = tgt_img[:, :3, :, :].to(device)
        refs_d = [img[:, :3, :, :].to(device) for img in ref_imgs]
        intr   = intrinsics.to(device)

        tgt_depth  = _depth_scales(disp_net(tgt_d))
        ref_depths = [_depth_scales(disp_net(r)) for r in refs_d]
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img.to(device), [img.to(device) for img in ref_imgs])

        # eval mode yields a single scale; score the loss on what's available
        n_scales = min(args.num_scales, len(tgt_depth))
        loss_photo, _ = compute_photo_and_geometry_loss(
            tgt_d, refs_d, intr,
            tgt_depth, ref_depths, poses, poses_inv,
            n_scales, args.with_ssim,
            args.with_mask, args.with_auto_mask, args.padding_mode)

    return loss_photo.item()


def probe_ratio_loss(args, tgt_img, ref_imgs, intrinsics, disp_net, pose_net) -> float:
    """Contrast-normalized probe: photometric error under the predicted pose
    divided by the error of assuming zero motion (identity warp).

    Raw photometric loss cannot detect weather shift: fog lowers global image
    contrast, which lowers reconstruction error even while the pose collapses
    (vKITTI Scene01: fog probe mean 0.83 vs clone 0.94, t_err 3x worse). The
    zero-motion baseline drops by the same contrast factor, so their ratio
    cancels it and measures the fraction of frame-to-frame change the pose
    explains. In-domain values sit well below 1; values approaching 1 mean the
    pose is no better than not moving at all.

    Masks are deliberately NOT the training auto-mask: with a zero pose the
    warped image equals the reference, which empties the auto-mask and zeroes
    the baseline. Only the warp validity mask is applied, to both terms.
    """
    disp_net.eval()
    pose_net.eval()

    with torch.no_grad():
        tgt_d  = tgt_img[:, :3, :, :].to(device)
        refs_d = [img[:, :3, :, :].to(device) for img in ref_imgs]
        intr   = intrinsics.to(device)

        disp = disp_net(tgt_d)
        tgt_depth = 1 / (disp[0] if isinstance(disp, (list, tuple)) else disp)

        num = 0.0
        den = 0.0
        for ref_full, ref in zip(ref_imgs, refs_d):
            rdisp = disp_net(ref)
            ref_depth = 1 / (rdisp[0] if isinstance(rdisp, (list, tuple)) else rdisp)
            pose = pose_net(tgt_img.to(device), ref_full.to(device))

            ref_warped, valid_mask, _, _ = inverse_warp2(
                ref, tgt_depth, ref_depth, pose, intr, args.padding_mode)

            diff_warp   = (tgt_d - ref_warped).abs().clamp(0, 1)
            diff_static = (tgt_d - ref).abs().clamp(0, 1)
            if args.with_ssim:
                diff_warp   = 0.15 * diff_warp   + 0.85 * compute_ssim_loss(tgt_d, ref_warped)
                diff_static = 0.15 * diff_static + 0.85 * compute_ssim_loss(tgt_d, ref)

            mask = valid_mask.expand_as(diff_warp)
            norm = mask.sum().clamp(min=1)
            num += (diff_warp * mask).sum() / norm
            den += (diff_static * mask).sum() / norm

    # The static-frame gate quantity is den normalized by the image's own
    # contrast: raw den scales with atmospheric transmission (fog shrinks
    # it ~3x, which read as "static" and suppressed every fog fire in
    # simulation), while den/contrast cancels that factor — moving frames
    # sit at ~0.2-0.5 in every weather, standstill at ~0.02-0.07.
    den_gate = den / (tgt_d.std() + 1e-12)
    return (num / (den + 1e-12)).item(), den_gate.item()


def probe_bnstats_distance(pose_net, tgt_img, ref_img) -> float:
    """Feature-statistics probe: how far the current frames' activation
    statistics sit from the training domain, measured at every BatchNorm layer
    of pose_net as mean over channels of |batch mean - running mean| / sqrt(
    running var). Absolute with respect to the training domain (no in-stream
    calibration needed) and non-photometric, so immune to the contrast
    confound that breaks the raw loss probe under fog.
    """
    dists = []
    hooks = []

    def hook(module, inputs, output):
        x = inputs[0]
        mu_b = x.mean(dim=(0, 2, 3))
        d = ((mu_b - module.running_mean).abs()
             / (module.running_var + module.eps).sqrt()).mean()
        dists.append(d.item())

    for m in pose_net.modules():
        if isinstance(m, torch.nn.BatchNorm2d) and m.track_running_stats:
            hooks.append(m.register_forward_hook(hook))

    pose_net.eval()
    with torch.no_grad():
        pose_net(tgt_img.to(device), ref_img.to(device))

    for h in hooks:
        h.remove()
    return float(np.mean(dists))


def zero_motion_baseline(args, tgt_img, ref_imgs) -> float:
    """Photometric cost of assuming no motion at all: the raw frame-to-frame
    difference, averaged over the reference frames. Pure image arithmetic (no
    network passes). Under the atmospheric scattering model both this baseline
    and the warped photometric loss shrink by the same transmission factor, so
    dividing by it makes the adaptation objective contrast-invariant."""
    with torch.no_grad():
        tgt = tgt_img[:, :3, :, :].to(device)
        vals = []
        for ref_full in ref_imgs:
            ref  = ref_full[:, :3, :, :].to(device)
            diff = (tgt - ref).abs().clamp(0, 1)
            if args.with_ssim:
                diff = 0.15 * diff + 0.85 * compute_ssim_loss(tgt, ref)
            vals.append(diff.mean())
        return float(torch.stack(vals).mean())


def compute_probe_signal(args, tgt_img, ref_imgs, intrinsics, disp_net, pose_net):
    """Returns (signal, zero_motion_den). den is None for signals that have
    no denominator; for 'ratio' it feeds the static-frame gate."""
    if args.probe_signal == 'ratio':
        return probe_ratio_loss(args, tgt_img, ref_imgs, intrinsics, disp_net, pose_net)
    if args.probe_signal == 'bnstats':
        return probe_bnstats_distance(pose_net, tgt_img, ref_imgs[0]), None
    return probe_photo_loss(args, tgt_img, ref_imgs, intrinsics, disp_net, pose_net), None


def main():
    args     = parser.parse_args()
    use_lora = args.lora_rank > 0

    # ── Load pretrained models ────────────────────────────────────────────────
    pose_net, disp_net, weights_pose, weights_disp = init_models(args, device)

    # ── Optionally inject LoRA adapters ───────────────────────────────────────
    if use_lora:
        # Step 1: freeze everything in both networks
        freeze_all(pose_net)
        freeze_all(disp_net)   # disp_net stays frozen throughout the whole sequence

        # Step 2: inject low-rank adapters into the chosen modules of pose_net.
        # This un-freezes only the newly added LoRA parameters (lora_A, lora_B,
        # lora_down, lora_up) while keeping the original weights frozen.
        inject_lora_pose_net(pose_net,
                             rank=args.lora_rank,
                             alpha=args.lora_alpha,
                             targets=args.lora_targets)

        n_lora  = count_lora_params(pose_net)
        n_total = sum(p.numel() for p in pose_net.parameters())
        print(f'LoRA enabled  |  targets={args.lora_targets}  rank={args.lora_rank}  '
              f'alpha={args.lora_alpha}')
        print(f'Trainable params: {n_lora:,} / {n_total:,} '
              f'({100 * n_lora / n_total:.3f}% of pose_net)')

    elif args.part:
        freeze_except_decoder(pose_net)
        freeze_except_decoder(disp_net)

    # ── Initialise "best weights" bookkeeping ─────────────────────────────────
    # These accumulate across the whole sequence: whenever a frame produces a
    # lower photo loss than any previous frame, the current weights are saved.
    weights_disp_best = weights_disp['state_dict']
    if use_lora:
        # LoRA adapters start at zero correction (B=0), so the initial snapshot
        # is just the zero-adapter state.
        weights_pose_best = lora_state_dict(pose_net)
    else:
        weights_pose_best = weights_pose['state_dict']

    print('=> setting adam solver')
    optimizer = build_optimizer(args, pose_net, disp_net, use_lora=use_lora)

    # ── Retrieve image files ──────────────────────────────────────────────────
    image_dir  = Path(args.dataset_dir + args.sequence + "/image_2/")
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()
    test_files = sum([image_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])
    test_files.sort()
    n = len(test_files)
    print(f'{n} files to test')

    global_pose = np.eye(4)
    poses       = [global_pose[0:3, :].reshape(1, 12)]

    intrinsics = torch.tensor(
        np.genfromtxt(image_dir / 'cam.txt').astype(np.float32).reshape((3, 3)))
    intrinsics = torch.unsqueeze(intrinsics, 0)

    if args.thread:
        print('loading imgs...')
        tensor_imgs_total = preload_images(test_files, args)

    # ── Trigger state ─────────────────────────────────────────────────────────
    # ema_loss: exponential moving average of per-frame photometric loss.
    # Updated at every frame (whether we adapt or not) to track the "normal"
    # difficulty level of the current sequence.
    ema_loss     = -1.0
    adapt_count  = 0
    skip_count   = 0
    probe_losses = []   # per-frame probe loss trace (probe-only / trigger runs)

    # ── CUSUM trigger state ───────────────────────────────────────────────────
    cusum_S     = 0.0
    cusum_mu    = args.cusum_ref_mean
    cusum_sigma = args.cusum_ref_std
    cusum_calib = []    # losses collected during start-of-sequence calibration

    # ── Recovery state (undoable adaptation) ──────────────────────────────────
    shifted        = False   # a CUSUM fire has happened; adapters may be dirty
    recovery_count = 0       # consecutive frames the BASE model was in-band
    events         = []      # (frame, 'adapt'|'reset', signal) — saved at exit

    # ── Static-frame gate state (ratio signal only) ───────────────────────────
    den_hist = []            # recent zero-motion baselines; median = "moving" level

    # ── Main inference loop ───────────────────────────────────────────────────
    for iter in tqdm(range(n - int(args.sequence_length) + 1)):

        tensor_imgs = []
        if args.thread:
            for i in range(args.sequence_length):
                tensor_imgs.append(tensor_imgs_total[iter + i].to(device))
        else:
            for i in range(args.sequence_length):
                tensor_imgs.append(load_tensor_image(test_files[iter + i], args).to(device))

        tgt_img  = tensor_imgs[0]
        ref_imgs = tensor_imgs[1:]

        tgt_img_forDepth = tgt_img[:, :3, :, :]

        # ── Trigger: decide whether this frame needs adaptation ───────────────
        # If --adapt-threshold is 0, do_adapt is always True (original behaviour).
        # --probe-only records the per-frame loss WITHOUT ever adapting, so
        # trigger designs can be simulated offline on the recorded trace.
        do_adapt = True
        static_frame = False
        if args.probe_only or args.adapt_threshold > 0 or args.cusum_h > 0:
            current_loss, zero_den = compute_probe_signal(args, tgt_img, ref_imgs, intrinsics, disp_net, pose_net)
            probe_losses.append(current_loss)

            # ── Static-frame gate: at (near-)standstill the ratio divides two
            # near-zero losses and explodes; the frame also carries no motion
            # information worth adapting on. Hold all trigger state.
            if zero_den is not None and args.static_floor > 0:
                if len(den_hist) >= 20:
                    med = sorted(den_hist)[len(den_hist) // 2]
                    static_frame = zero_den < args.static_floor * med
                if not static_frame:
                    den_hist.append(zero_den)
                    if len(den_hist) > 100:
                        den_hist.pop(0)

        if static_frame:
            do_adapt    = False
            skip_count += 1
        elif args.probe_only:
            do_adapt    = False
            skip_count += 1
        elif args.cusum_h > 0:
            # ── CUSUM trigger: adapt only on SUSTAINED loss elevation ──────────
            # S accumulates evidence that the loss mean has shifted above the
            # reference; transient hard frames (turns, glare) drain back out via
            # the max(0, .) floor. Fires only when elevation persists = domain
            # shift. Chosen over the spike rule after offline simulation on
            # recorded traces (see trigger_sim.py): the spike rule fired at the
            # hardest frames, which measurably degraded accuracy.
            if cusum_mu is None:
                # start-of-sequence calibration (assumes in-domain start)
                cusum_calib.append(current_loss)
                do_adapt    = False
                skip_count += 1
                if len(cusum_calib) >= args.cusum_calib_frames:
                    arr         = np.asarray(cusum_calib)
                    cusum_mu    = float(arr.mean())
                    cusum_sigma = float(arr.std()) + 1e-12
                    print(f'CUSUM calibrated: mu={cusum_mu:.4f} sigma={cusum_sigma:.4f}')
            else:
                cusum_S = max(0.0, cusum_S + (current_loss - cusum_mu
                                              - args.cusum_kappa * cusum_sigma))
                if cusum_S > args.cusum_h * cusum_sigma:
                    cusum_S = 0.0   # reset: adaptation happens this frame
                    shifted = True
                    events.append((iter, 'adapt', current_loss))
                else:
                    do_adapt    = False
                    skip_count += 1
        elif args.adapt_threshold > 0:
            # Warm up EMA on the first frame, then update with decay 0.9
            if ema_loss < 0:
                ema_loss = current_loss
            else:
                ema_loss = 0.9 * ema_loss + 0.1 * current_loss

            # Adapt only on SURPRISE: loss spiking above the recent moving
            # average signals unfamiliar conditions (domain shift). On
            # stationary in-domain footage current_loss ~= ema_loss, so the
            # trigger stays quiet and we keep the offline weights.
            # (The previous form skipped only on unusually EASY frames, which
            # almost never occur — measured 0% skips, making D identical to C.)
            if current_loss <= args.adapt_threshold * ema_loss:
                do_adapt    = False
                skip_count += 1

        # ── Recovery detection: undo the adaptation once the shift passes ─────
        # While shifted, also probe with adapters DISABLED and compare: reset
        # once the pristine base model matches/beats the adapted model for
        # --recovery-patience consecutive frames, i.e. the adapters have
        # stopped helping on the current world. Relative, so it needs no
        # reference band and is immune to section difficulty (an absolute
        # in-band test proved brittle: hard road sections sit at the band edge
        # and break any consecutive count). Both probes see the same frames,
        # so their fluctuations cancel in the comparison. Probing the adapted
        # model alone would oscillate: adapt -> signal drops -> reset ->
        # signal spikes -> adapt ...
        if (args.reset_on_recovery and use_lora and shifted and not static_frame
                and args.cusum_h > 0 and cusum_mu is not None):
            set_lora_enabled(pose_net, False)
            base_signal, _ = compute_probe_signal(args, tgt_img, ref_imgs, intrinsics, disp_net, pose_net)
            set_lora_enabled(pose_net, True)
            # current_loss is the adapted-model probe. The adapter must beat
            # the base by a significant margin to stay; ties and dust-level
            # wins count toward the reset.
            if base_signal <= current_loss + args.recovery_margin * cusum_sigma:
                recovery_count += 1
            else:
                recovery_count = 0
            if recovery_count >= args.recovery_patience:
                reset_lora(pose_net)
                weights_pose_best = lora_state_dict(pose_net)
                optimizer = build_optimizer(args, pose_net, disp_net, use_lora=use_lora)
                cusum_S, shifted, recovery_count = 0.0, False, 0
                events.append((iter, 'reset', base_signal))
                tqdm.write(f'[frame {iter}] shift over -> adapters reset to pristine base model')

        # ── Adaptation step (skipped on "easy" frames when trigger is active) ─
        if do_adapt:
            adapt_count += 1
            photo_norm = (zero_motion_baseline(args, tgt_img, ref_imgs)
                          if args.adapt_loss == 'norm' else None)
            errors, error_names, weights_disp_best, weights_pose_best = train(
                args, tgt_img, ref_imgs,
                intrinsics=intrinsics,
                disp_net=disp_net,
                pose_net=pose_net,
                optimizer=optimizer,
                epochs=args.epochs,
                weights_disp_best=weights_disp_best,
                weights_pose_best=weights_pose_best,
                use_lora=use_lora,
                photo_norm=photo_norm)

            if args.select_best:
                if use_lora:
                    # Restore pose_net's LoRA adapters to the best snapshot.
                    # Cost: copy tens of KB — not ~150 MB.
                    load_lora_state_dict(pose_net, weights_pose_best)
                    # disp_net: untouched (fully frozen, no state change)
                else:
                    pose_net.load_state_dict(weights_pose_best, strict=True)
                    disp_net.load_state_dict(weights_disp_best, strict=True)
                    if args.part:
                        freeze_except_decoder(pose_net)
                        freeze_except_decoder(disp_net)

                # Rebuild optimizer to reset Adam momentum after weight restoration.
                # With LoRA this is cheap (few parameters); with full adaptation it
                # was already done in the original code.
                optimizer = build_optimizer(args, pose_net, disp_net, use_lora=use_lora)

        # ── Pose inference for this frame ─────────────────────────────────────
        pose        = pose_net(tgt_img, ref_imgs[0])
        global_pose = update_global_pose(global_pose, pose)
        poses.append(global_pose[0:3, :].reshape(1, 12))

        # ── Optional depth/disparity output ──────────────────────────────────
        if args.output_disp or args.output_depth:
            output    = disp_net(tgt_img_forDepth)[0]
            file_name = 'depth/' + str(iter)

            if args.output_disp:
                disp = (255 * tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
                imsave(output_dir / '{}_disp{}'.format(file_name, ".png"),
                       np.transpose(disp, (1, 2, 0)))
            if args.output_depth:
                depth = 1 / output
                depth = (255 * tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
                imsave(output_dir / '{}_depth{}'.format(file_name, ".png"),
                       np.transpose(depth, (1, 2, 0)))

        if iter == n - args.sequence_length:
            global_pose, poses = process_tail_poses(
                pose_net, test_files,
                n - args.sequence_length + 1,
                args.sequence_length,
                global_pose, poses, device, args)

    total = adapt_count + skip_count
    print(f'\nAdaptation summary: {adapt_count} adapted, {skip_count} skipped '
          f'({100 * skip_count / max(total, 1):.1f}% skipped)')

    if probe_losses:
        loss_file = Path(args.output_dir + 'probe_losses_' + args.sequence + '.txt')
        np.savetxt(loss_file, np.asarray(probe_losses), fmt='%1.6e')
        print(f'Probe loss trace ({len(probe_losses)} frames) -> {loss_file}')

    if events:
        ev_file = Path(args.output_dir + 'trigger_events_' + args.sequence + '.txt')
        with open(ev_file, 'w') as f:
            for frame, kind, val in events:
                f.write(f'{frame} {kind} {val:.6f}\n')
        print(f'{len(events)} trigger events -> {ev_file}')

    poses    = np.concatenate(poses, axis=0)
    filename = Path(args.output_dir + args.sequence + ".txt")
    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')


if __name__ == '__main__':
    main()
