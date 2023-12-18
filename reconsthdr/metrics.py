from typing import List

import numpy as np

L1_ERROR_KEY = "L1_error"

def calc_metrics(net_output: np.ndarray, batch: np.ndarray) -> List[float]:
    metrics = {}

def l1_error(pred: np.ndarray, gt: np.ndarray) -> float:
    return np.abs(pred - gt).mean()

metrics_dict = {
    L1_ERROR_KEY: l1_error
}
