from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

from reconsthdr.predictor import Predictor
from reconsthdr.utils import save_hdr


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--i', help="file or directory of input LDR img", type=str, default="data/PolyHeven/ldr")
    parser.add_argument('--cfg', help="path to the config file", type=str, default="config/config.yaml")
    parser.add_argument('--weight', help="path to the weight file", type=str, default="best-val-loss-epoch297.pth")
    parser.add_argument('--save_dir', help="path to save predicted results", type=str, default=None)
    parser.add_argument('--out_height', help="height of the output hdr image", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    predictor = Predictor(args.cfg, args.weight)
    input_path = Path(args.i)
    save_dir = Path(args.save_dir) if args.save_dir is not None else input_path if input_path.is_dir() else input_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    for ldr_img_file in input_path.glob("*") if input_path.is_dir() else [input_path]:
        input_img = np.array(Image.open(ldr_img_file).resize((args.out_height*2, args.out_height))).astype(np.float32) / 255.
        output_img = predictor(input_img)
        save_hdr(save_dir / f"{ldr_img_file.stem}.hdr", output_img)
