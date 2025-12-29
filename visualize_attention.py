import os
import torch
import imageio
import models
import cv2
import numpy as np
import argparse
from imageio import imread
from skimage.transform import resize as imresize
from skimage import img_as_ubyte
from tqdm import tqdm
from inverse_warp import *
from utils import extract_orb


parser = argparse.ArgumentParser(description='Script for visualizing attention weights',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--frame_folder", type=str, required=True, help="Path to the folder containing frames.")
parser.add_argument("--output_dir", type=str, default="attn_weights", help="Path to the output directory (default: attn_weights).",)
parser.add_argument("--weights", type=str, required=True, help="Path to the posenet checkpoints")

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

@torch.no_grad()
def main():
    
    args = parser.parse_args()          
    os.makedirs(args.output_dir, exist_ok=True)

    weights_pose = torch.load(args.weights)
    pose_net = models.PoseResNet().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net.eval()

    frame_files = sorted([os.path.join(args.frame_folder, f) for f in os.listdir(args.frame_folder) if f.endswith(('.png', '.jpg'))])

    img_width, img_height = 832, 256

    for frame_idx in tqdm(range(len(frame_files) - 1)):
        img1_path = frame_files[frame_idx]
        img2_path = frame_files[frame_idx + 1]
        
        tensor_img1 = load_tensor_image(img1_path, args).to(device)
        tensor_img2 = load_tensor_image(img2_path, args).to(device)

        pose, att_weights = pose_net(tensor_img1, tensor_img2, True)

        original_image = cv2.imread(img1_path)
        original_image = cv2.resize(original_image, (img_width, img_height))

        att_weights = att_weights.squeeze(0).cpu().numpy()
        avg_att_weights = np.expand_dims(att_weights.mean(0), axis=0)
        att_weights = np.concatenate((att_weights, avg_att_weights), axis=0)

        for head in range(att_weights.shape[0]):
            att_map = att_weights[head].T.mean(-1).reshape(8, 26)
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())

            att_map_resized = cv2.resize(att_map, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            heatmap = cv2.applyColorMap(np.uint8(255 * att_map_resized), cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
            if head+1 != att_weights.shape[0]:
                output_dir_frame = os.path.join(args.output_dir, f'head{head+1}')
            else:
                output_dir_frame = os.path.join(args.output_dir, f'average')
            os.makedirs(output_dir_frame, exist_ok=True)

            save_path = os.path.join(output_dir_frame, f'frame_{frame_idx+1:04d}_head_{head+1}.png')
            cv2.imwrite(save_path, superimposed_img)
            # print(f'Saved attention map for head {head+1} on frame {frame_idx+1} to {save_path}')

    for head in range(att_weights.shape[0]):
        frames = []
        if head+1 != att_weights.shape[0]:
            output_dir_frame = os.path.join(args.output_dir, f'head{head+1}')
            gif_path = os.path.join(args.output_dir, f'head{head+1}.gif')
        else:
            output_dir_frame = os.path.join(args.output_dir, f'average')
            gif_path = os.path.join(args.output_dir, f'average.gif')
            
        for frame_idx in tqdm(range(len(frame_files) - 1)):
            img_path = os.path.join(output_dir_frame, f'frame_{frame_idx+1:04d}_head_{head+1}.png')
            frame = cv2.imread(img_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (416, 128), interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)
            
        imageio.mimsave(gif_path, frames, duration=0.05)

        print(f'Saved GIF for head {head+1} to {gif_path}')


if __name__ == '__main__':
    
    main()