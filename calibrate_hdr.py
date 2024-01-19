"""
Calibrate HDRIs with LDRIs by following the procedure in the paper 'Luminance Attentive Networks for HDR Image and Panorama Reconstruction' (https://arxiv.org/abs/2109.06688)
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from reconsthdr.dataset.tone_mappers import calibrate_hdr
from reconsthdr.utils import load_hdr


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--ldr_in_dir', help="directory of input LDRIs", type=str, default="data/PolyHeven/ldr")
    parser.add_argument('--hdr_in_dir', help="directory of input HDRIs", type=str, default="data/PolyHeven/hdr")
    parser.add_argument('--ldr_out_dir', help="directory to save LDRIs", type=str, default="data/PolyHeven/calibrated_ldr")
    parser.add_argument('--hdr_out_dir', help="directory to save calibrated HDRIs", type=str, default="data/PolyHeven/calibrated_hdr")
    return parser.parse_args()

def main(args):
    hdr_in_dir = Path(args.hdr_in_dir)
    ldr_in_dir = Path(args.ldr_in_dir)
    ldr_out_dir = Path(args.ldr_out_dir)
    hdr_out_dir = Path(args.hdr_out_dir)
    ldr_out_dir.mkdir(parents=True, exist_ok=True)
    hdr_out_dir.mkdir(parents=True, exist_ok=True)
    for hdr_file in hdr_in_dir.glob("*.hdr"):
        hdr_file_id = f"{'_'.join(hdr_file.stem.split('_')[:-1])}"  # input hdr_file name example: abandoned_factory_canteen_01_2k.hdr
        ldr_file = ldr_in_dir / f"{hdr_file_id}.jpg"
        if not ldr_file.exists():
            print(ldr_file)
            continue
        hdr_img = load_hdr(hdr_file)
        ldr_img = np.array(Image.open(ldr_file))

        hdr_img, ldr_img = match_hdr_ldr_size(hdr_img, ldr_img)
        calibrated_hdr, calibrated_ldr = calibrate_hdr(hdr_img, ldr_img)
        
        cv2.imwrite(str(hdr_out_dir / f"{hdr_file_id}.hdr"), calibrated_hdr)
        cv2.imwrite(str(ldr_out_dir / f"{hdr_file_id}.hdr"), calibrated_ldr)

def match_hdr_ldr_size(hdr: np.ndarray, ldr: np.ndarray):
    if hdr.shape[:2] != ldr.shape[:2]:
        smaller_shape = np.min([hdr.shape[:2], ldr.shape[:2]], axis=0)
        hdr_resized = cv2.resize(hdr, (smaller_shape[1], smaller_shape[0]))
        ldr_resized = cv2.resize(ldr, (smaller_shape[1], smaller_shape[0]))
        return hdr_resized, ldr_resized
    return hdr, ldr


if __name__ == "__main__":
    main(get_args())
