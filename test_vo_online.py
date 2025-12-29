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
# for partial adaptation
parser.add_argument("--part", action='store_true', help="partial adaptation")
# for best selection
parser.add_argument("--select-best", type=int, default=1, help="select best parameters or not")

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
                # Update the progress bar when a future is completed
                pbar.update(1)

        # Retrieve results
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

def build_optimizer(args, pose_net, disp_net):
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    return torch.optim.Adam(
        optim_params,
        betas=(args.momentum, args.beta),
        weight_decay=args.weight_decay
    )

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

def train(args, tgt_img, ref_imgs, intrinsics, disp_net, pose_net, optimizer, epochs, weights_disp_best, weights_pose_best):
    global device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    disp_net.train()
    pose_net.train()

    end = time.time()
    best_error = -1

    for i in range(epochs):
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)

        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        tgt_img_forDepth = tgt_img[:,:3,:,:]
        ref_imgs_forDepth = [img[:,:3,:,:] for img in ref_imgs]

        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img_forDepth, ref_imgs_forDepth)
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img_forDepth, ref_imgs_forDepth, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img_forDepth, ref_depths, ref_imgs_forDepth)

        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

        losses.update([loss, loss_1, loss_2, loss_3])

        decisive_error = losses.val[1]# choose the metric

        # first round
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error #only store if error is lower than last round，won't store if epoch set to 1
        best_error = min(best_error, decisive_error)

        if is_best:

            weights_disp_best = copy.deepcopy(disp_net.state_dict())
            weights_pose_best = copy.deepcopy(pose_net.state_dict())
        
        # compute gradient and do Adam step
        if i<epochs-1 or epochs == 1 or args.select_best==0:
            optimizer.zero_grad() # this can be put before is_best to update the weight for one more time
            loss.backward()
            optimizer.step()

    # print('losses.avg[0]',losses.avg[0])
    return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss'], weights_disp_best, weights_pose_best

def main():
    args = parser.parse_args()

    # Initialize models and optimzer
    pose_net, disp_net, weights_pose, weights_disp = init_models(args, device)
    if args.part:
        freeze_except_decoder(pose_net)
        freeze_except_decoder(disp_net)
    # best weights are initialized with pretrained weights
    weights_disp_best, weights_pose_best = weights_disp['state_dict'], weights_pose['state_dict']
    print('=> setting adam solver')
    optimizer = build_optimizer(args, pose_net, disp_net)


    # Retrieve files
    image_dir = Path(args.dataset_dir + args.sequence + "/image_2/")
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()
    test_files = sum([image_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])
    test_files.sort()    
    n = len(test_files)
    print(f'{n} files to test')

    global_pose = np.eye(4)
    poses = [global_pose[0:3, :].reshape(1, 12)]

    intrinsics = torch.tensor(np.genfromtxt(image_dir/'cam.txt').astype(np.float32).reshape((3, 3)))
    intrinsics = torch.unsqueeze(intrinsics, 0)

    if args.thread:
        print('loading imgs...')
        tensor_imgs_total = preload_images(test_files, args)

    # Main inference loop with online adaptation
    for iter in tqdm(range(n - int(args.sequence_length) + 1)):
        tensor_imgs = []
        # print("iter",iter)
        if args.thread:
            for i in range(args.sequence_length):
                tensor_imgs.append(tensor_imgs_total[iter + i].to(device))
        else:
            for i in range(args.sequence_length):
                tensor_imgs.append(load_tensor_image(test_files[iter + i], args).to(device))

        tgt_img = tensor_imgs[0]
        ref_imgs = tensor_imgs[1:]

        tgt_img_forDepth = tgt_img[:,:3,:,:]
        # ref_imgs_forDepth = [img[:,:3,:,:] for img in ref_imgs]

        errors, error_names, weights_disp_best, weights_pose_best = train(args, tgt_img, ref_imgs, intrinsics=intrinsics,
                                             disp_net=disp_net,pose_net=pose_net, optimizer=optimizer, epochs=args.epochs,
                                            weights_disp_best=weights_disp_best, weights_pose_best=weights_pose_best)

        if args.select_best:
            
            pose_net.load_state_dict(weights_pose_best, strict=True)
            disp_net.load_state_dict(weights_disp_best, strict=True)
            if args.part:
                freeze_except_decoder(pose_net)
                freeze_except_decoder(disp_net)
            # necessary to re-initialize optimizer
            optimizer = build_optimizer(args, pose_net, disp_net)

        pose = pose_net(tgt_img, ref_imgs[0])
        global_pose = update_global_pose(global_pose, pose)

        poses.append(global_pose[0:3, :].reshape(1, 12))

        # depth
        if args.output_disp or args.output_depth:
            output = disp_net(tgt_img_forDepth)[0]
        # print(output)

        file_name = 'depth/'+str(iter)

        if args.output_disp:
            disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
            imsave(output_dir / '{}_disp{}'.format(file_name, ".png"), np.transpose(disp, (1, 2, 0)))#改png后才能保存成功，不然读jpg保存.jpg会失败，很奇怪，sfmlearner就没问题
        if args.output_depth:
            depth = 1/output
            depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
            imsave(output_dir/'{}_depth{}'.format(file_name, ".png"), np.transpose(depth, (1, 2, 0)))

        if iter == n - args.sequence_length:
            global_pose, poses = process_tail_poses(
                pose_net,
                test_files,
                n - args.sequence_length + 1,
                args.sequence_length,
                global_pose,
                poses,
                device,
                args
            )


    poses = np.concatenate(poses, axis=0)
    filename = Path(args.output_dir + args.sequence + ".txt")
    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')


if __name__ == '__main__':
    main()
