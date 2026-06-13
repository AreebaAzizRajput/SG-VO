import argparse
import time
import copy
import concurrent.futures

import torch
import numpy as np
from tqdm import tqdm
from imageio import imread, imsave
from skimage.transform import resize as imresize
from skimage import img_as_ubyte
from path import Path

from models import DispResNet, PoseResNet
from logger import AverageMeter
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss
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
parser.add_argument('--adapt-threshold', type=float, default=0.0,
                    help='Skip adaptation when photo_loss < adapt_threshold × EMA_loss. '
                         '0.0 = always adapt. Recommended starting value: 0.8.')


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
          weights_disp_best, weights_pose_best, use_lora=False):
    global device
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter(i=4, precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    disp_net.train()
    pose_net.train()

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
    with torch.no_grad():
        tgt_d  = tgt_img[:, :3, :, :].to(device)
        refs_d = [img[:, :3, :, :].to(device) for img in ref_imgs]
        intr   = intrinsics.to(device)

        tgt_depth, ref_depths = compute_depth(disp_net, tgt_d, refs_d)
        poses, poses_inv      = compute_pose_with_inv(pose_net, tgt_img.to(device), [img.to(device) for img in ref_imgs])

        loss_photo, _ = compute_photo_and_geometry_loss(
            tgt_d, refs_d, intr,
            tgt_depth, ref_depths, poses, poses_inv,
            args.num_scales, args.with_ssim,
            args.with_mask, args.with_auto_mask, args.padding_mode)

    return loss_photo.item()


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
    ema_loss    = -1.0
    adapt_count = 0
    skip_count  = 0

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
        do_adapt = True
        if args.adapt_threshold > 0:
            current_loss = probe_photo_loss(args, tgt_img, ref_imgs, intrinsics, disp_net, pose_net)

            # Warm up EMA on the first frame, then update with decay 0.9
            if ema_loss < 0:
                ema_loss = current_loss
            else:
                ema_loss = 0.9 * ema_loss + 0.1 * current_loss

            # Skip adaptation if current frame is "easy" relative to recent average
            if current_loss < args.adapt_threshold * ema_loss:
                do_adapt    = False
                skip_count += 1

        # ── Adaptation step (skipped on "easy" frames when trigger is active) ─
        if do_adapt:
            adapt_count += 1
            errors, error_names, weights_disp_best, weights_pose_best = train(
                args, tgt_img, ref_imgs,
                intrinsics=intrinsics,
                disp_net=disp_net,
                pose_net=pose_net,
                optimizer=optimizer,
                epochs=args.epochs,
                weights_disp_best=weights_disp_best,
                weights_pose_best=weights_pose_best,
                use_lora=use_lora)

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

    poses    = np.concatenate(poses, axis=0)
    filename = Path(args.output_dir + args.sequence + ".txt")
    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')


if __name__ == '__main__':
    main()
