"""
Calibrate HDRIs with LDRIs by following the procedure in the paper 'Luminance Attentive Networks for HDR Image and Panorama Reconstruction' (https://arxiv.org/abs/2109.06688)
"""

from argparse import ArgumentParser
from pathlib import Path

from reconsthdr.utils import (load_rgb_hdr, load_rgb_ldr, match_image_size,
                              save_hdr, save_ldr)
from reconsthdr.utils.hdr_calibration import calibrate_hdr


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
        hdr_img = load_rgb_hdr(hdr_file)
        ldr_img = load_rgb_ldr(ldr_file)

        hdr_img, ldr_img = match_image_size(hdr_img, ldr_img)
        calibrated_hdr = calibrate_hdr(hdr_img, ldr_img)
        
        save_hdr(str(hdr_out_dir / f"{hdr_file_id}.hdr"), calibrated_hdr)
        save_ldr(str(ldr_out_dir / f"{hdr_file_id}.jpg"), ldr_img)


if __name__ == "__main__":
    main(get_args())
