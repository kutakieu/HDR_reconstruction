from typing import Dict

import numpy as np
from lightning import pytorch as pl

from reconsthdr.lossfn_optimizer import optimizer_factory
from reconsthdr.metrics import metrics_dict
from reconsthdr.models import loss_fn_factory, model_factory
from reconsthdr.utils.logger import get_logger

logger = get_logger(__name__)


class LightningWrapper(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.net = model_factory(cfg)
        self.loss_fn = loss_fn_factory(cfg)
        self.optimizer_name = cfg.training.optimizer.name
        self.learning_rate = cfg.training.optimizer.learning_rate
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    # override
    def configure_optimizers(self):
        optimizer = optimizer_factory(self.optimizer_name, self.learning_rate, self.net)
        return optimizer

    # override
    def training_step(self, batch, batch_idx):
        pred = self.net(batch)
        loss = self.loss_fn(pred, batch)
        self.log("loss_train_step", loss["total"], sync_dist=True)
        self.training_step_outputs.append({"loss": loss["total"].detach().cpu().numpy()})
        return loss["total"]

    # override
    def on_train_epoch_end(self) -> None:
        loss_avg = np.stack([x["loss"] for x in self.training_step_outputs]).mean()
        self.log("loss_train_epoch", loss_avg, sync_dist=True)
        self.training_step_outputs.clear()

    # override
    def validation_step(self, batch, batch_idx):
        loss, net_output = self.feed_forward_fn(self.net, batch)
        metrics = self.calc_metrics(net_output, batch)
        self.validation_step_outputs.append({"loss": loss["total"].detach().cpu().numpy(), "metrics": metrics})
        return loss["total"]

    # override
    def on_validation_epoch_end(self) -> None:
        loss_avg = np.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("loss_val", loss_avg, sync_dist=True)
        for key in metrics_dict.keys():
            iou_avg = np.stack([x["metrics"][key] for x in self.validation_step_outputs]).mean()
            self.log(f"{key}_val", iou_avg, sync_dist=True)
        self.validation_step_outputs.clear()

    # override
    def test_step(self, batch, batch_idx):
        loss, net_output = self.feed_forward_fn(self.net, batch)
        metrics = self.calc_metrics(net_output, batch)
        self.test_step_outputs.append({"loss": loss["total"].detach().cpu().numpy(), "metrics": metrics})
        return loss["total"]

    # override
    def on_test_epoch_end(self) -> None:
        loss_avg = np.stack([x["loss"] for x in self.test_step_outputs]).mean()
        self.log("loss_test", loss_avg, sync_dist=True)
        for key in metrics_dict.keys():
            iou_avg = np.stack([x["metrics"][key] for x in self.test_step_outputs]).mean()
            self.log(f"{key}_test", iou_avg, sync_dist=True)
        self.test_step_outputs.clear()

    def calc_metrics(self, pred_batch, gt_batch) -> Dict[str, float]:
        averaged_metrics = {}
        for key, metric_fn in metrics_dict.items():
            averaged_metrics[key] = metric_fn(pred_batch, gt_batch)
        return averaged_metrics
