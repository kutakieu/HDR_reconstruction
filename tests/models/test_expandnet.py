from pathlib import Path

import cv2
import pytest
import torch

from src.models.expandnet import ExpandNet

test_data_dir = Path("tests/data")

@pytest.mark.parametrize(
    "img_file",
    [test_data_dir / "brown_photostudio_02_0.5k.hdr"]
)
def test_expandnet(img_file):
    img = cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2RGB)
    img_tensor = torch.FloatTensor(img.transpose([2, 0, 1]))  # img_tensor shape: 3 x H x W
    print(img_tensor.shape)
    model = ExpandNet()
    model.eval()
    with torch.no_grad(): 
        pred = model(img_tensor.unsqueeze(0))
    assert pred.shape == (1, 3, 512, 512)
    
