#!/usr/bin/env python3
"""Compose a weather-switch sequence from two vKITTI conditions.

All vKITTI 2 conditions of a scene share the SAME camera trajectory, so
frames from different conditions can be spliced by index and the ground-truth
poses stay exactly valid. The result is a physically consistent drive through
a weather change (e.g. clear -> fog -> clear): the experiment that shows the
trigger firing on entry and the recovery reset firing after the shift passes.

Usage:
    python scripts/make_weather_splice.py \
        --base-dir data/vkitti/clone/sequences \
        --shift-dir data/vkitti/fog/sequences \
        --out-dir data/vkitti/splicefog/sequences \
        --sequence 01 --shift-start 0.33 --shift-end 0.67

--shift-start/--shift-end take fractions of the sequence (0..1) or absolute
frame indices (>1). Frames in [start, end) come from the shift condition,
the rest from the base condition. cam.txt is copied from the base condition.
"""
import argparse
import glob
import os
import shutil

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--base-dir', required=True,
                    help='sequences dir of the base condition (e.g. clone)')
parser.add_argument('--shift-dir', required=True,
                    help='sequences dir of the shifted condition (e.g. fog)')
parser.add_argument('--out-dir', required=True,
                    help='sequences dir to create for the spliced condition')
parser.add_argument('--sequence', required=True, help='sequence id, e.g. 01')
parser.add_argument('--shift-start', type=float, default=0.33,
                    help='start of the shifted segment (fraction <=1 or frame index)')
parser.add_argument('--shift-end', type=float, default=0.67,
                    help='end of the shifted segment (fraction <=1 or frame index)')


def main():
    args = parser.parse_args()
    base  = os.path.join(args.base_dir,  args.sequence, 'image_2')
    shift = os.path.join(args.shift_dir, args.sequence, 'image_2')
    out   = os.path.join(args.out_dir,   args.sequence, 'image_2')

    base_frames  = sorted(glob.glob(os.path.join(base,  '*.jpg')) +
                          glob.glob(os.path.join(base,  '*.png')))
    shift_frames = sorted(glob.glob(os.path.join(shift, '*.jpg')) +
                          glob.glob(os.path.join(shift, '*.png')))
    n = len(base_frames)
    assert n > 0, f'no frames in {base}'
    assert len(shift_frames) == n, (
        f'frame count mismatch: {n} base vs {len(shift_frames)} shift — '
        'conditions of the same scene must align frame-for-frame')

    s = int(args.shift_start * n) if args.shift_start <= 1 else int(args.shift_start)
    e = int(args.shift_end   * n) if args.shift_end   <= 1 else int(args.shift_end)
    assert 0 <= s < e <= n, f'bad shift window [{s}, {e}) for {n} frames'

    os.makedirs(out, exist_ok=True)
    for i in range(n):
        src = shift_frames[i] if s <= i < e else base_frames[i]
        dst = os.path.join(out, os.path.basename(base_frames[i]))
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    cam = os.path.join(base, 'cam.txt')
    if os.path.exists(cam):
        shutil.copy(cam, os.path.join(out, 'cam.txt'))

    print(f'splice ready: {out}  ({n} frames; shift segment = frames {s}-{e - 1})')
    print(f'GT poses of sequence {args.sequence} remain valid (shared trajectory).')


if __name__ == '__main__':
    main()
