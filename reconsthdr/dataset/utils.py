from pathlib import Path
from typing import Union

import cv2


def load_hdr(img_file: Union[str, Path]):
    return cv2.imread(str(img_file), flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
