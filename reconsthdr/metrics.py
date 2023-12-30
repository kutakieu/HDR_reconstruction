import numpy as np

L1_ERROR_KEY = "L1_error"


def l1_error(pred: np.ndarray, gt: np.ndarray) -> float:
    return np.mean(np.abs(np.exp(pred) - np.exp(gt)))

metrics_dict = {
    L1_ERROR_KEY: l1_error
}
