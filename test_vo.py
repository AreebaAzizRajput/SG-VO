import argparse
import concurrent.futures

import torch
import numpy as np
from tqdm import tqdm
from path import Path
from imageio import imread, imsave
from skimage.transform import resize as imresize
from skimage import img_as_ubyte

import models
from utils import extract_orb
from inverse_warp import pose_vec2mat

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", required=True, type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--thread", action="store_true", help="Preprocess images with multi-threads")

parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

parser.add_argument("--sequence", default='09', type=str, help="sequence to test")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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

def update_global_pose(global_pose, pose):
    pose_mat = pose_vec2mat(pose).squeeze(0).cpu().detach().numpy()
    pose_mat = np.vstack([pose_mat, [0, 0, 0, 1]])
    return global_pose @ np.linalg.inv(pose_mat)

def run_vo_sequence(
    test_files,
    pose_net,
    args,
    device
):
    global_pose = np.eye(4)
    poses = [global_pose[0:3, :].reshape(1, 12)]

    n = len(test_files)
    tensor_img1 = load_tensor_image(test_files[0], args).to(device)

    if args.thread:
        results = preload_images(test_files, args)

    for iter in tqdm(range(n - 1)):
        if args.thread:
            tensor_img2 = results[iter + 1].to(device)
        else:
            tensor_img2 = load_tensor_image(test_files[iter + 1], args).to(device)

        pose = pose_net(tensor_img1, tensor_img2)
        global_pose = update_global_pose(global_pose, pose)
        poses.append(global_pose[0:3, :].reshape(1, 12))
        tensor_img1 = tensor_img2

    poses = np.concatenate(poses, axis=0)
    return poses


@torch.no_grad()
def main():
    args = parser.parse_args()

    weights_pose = torch.load(args.pretrained_posenet)
    pose_net = models.PoseResNet().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=True)
    pose_net.eval()

    image_dir = Path(args.dataset_dir + args.sequence + "/image_2/")
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    test_files = sum([image_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])
    test_files.sort()

    print('{} files to test'.format(len(test_files)))

    poses = run_vo_sequence(
        test_files=test_files,
        pose_net=pose_net,
        args=args,
        device=device
    )

    filename = Path(args.output_dir + args.sequence + ".txt")
    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')


if __name__ == '__main__':
    main()
