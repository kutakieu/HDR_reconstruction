from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from reconsthdr.dataset.tone_mappers import calibrate_hdr
from reconsthdr.dataset.utils import load_hdr

if __name__ == "__main__":
    in_dir = Path("data/PolyHeven/hdr")
    ldr_out_dir = Path("data/PolyHeven/ldr")
    hdr_out_dir = Path("data/PolyHeven/calibrated_hdr")
    for hdr_file in in_dir.glob("*.hdr"):
        ldr_sample_file_name = f"{'_'.join(hdr_file.stem.split('_')[:-1])}"
        ldr_sample_file = in_dir / f"{ldr_sample_file_name}.png"
        hdr_img = load_hdr(hdr_file)
        ldr_sample_img = np.array(Image.open(ldr_sample_file))
        ldr_img = ldr_sample_img[:512, :, :3]
        Image.fromarray(ldr_img).save(ldr_out_dir / ldr_sample_file.name)
        calibrated_hdr = calibrate_hdr(cv2.resize(hdr_img, (1024, 512)), ldr_img)
        cv2.imwrite(str(hdr_out_dir / f"{ldr_sample_file_name}.hdr"), calibrated_hdr)
