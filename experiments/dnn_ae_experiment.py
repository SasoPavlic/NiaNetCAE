import torch
from lightning.pytorch import LightningModule
from torch import Tensor
from torch import optim
from torchmetrics import MeanSquaredError

from experiments.metrics import RMSE
from models.base import BaseAutoencoder


class DNNAEExperiment(LightningModule):
    def __init__(self,
                 dnn_ae_model: BaseAutoencoder,
                 params: dict,
                 n_features: int) -> None:
        super(DNNAEExperiment, self).__init__()

        # https://github.com/Lightning-AI/lightning/issues/4390#issuecomment-717447779
        # self.save_hyperparameters(logger=False)

        self.model = dnn_ae_model
        self.params = params
        self.n_features = n_features
        self.curr_device = None
        self.hold_graph = False
        # https://torchmetrics.readthedocs.io/en/latest/pages/overview.html#metrics-and-devices
        self.testing_RMSE_metric = RMSE()
        self.test_RMSE = None
        self.testing_MSE_metric = MeanSquaredError()
        self.test_RMSE = None

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = self.model.optimizer
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # TODO change training to convolutional
        real_signal = batch['image']
        labels = batch['depth']

        self.curr_device = real_signal.device
        results = self.forward(real_signal)
        train_loss = self.model.loss_function(*results,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True, on_step=False,
                      on_epoch=True)

        return train_loss['loss']

    def on_fit_end(self) -> None:
        self.test_model()
        self.test_RMSE = self.testing_RMSE_metric.compute()

    def test_model(self):

        dataloader_iterator = iter(self.trainer.datamodule.test_dataloader())

        while True:
            try:
                batch = next(dataloader_iterator)
                data = batch['image']
                target = batch['depth']
            except StopIteration:
                break
            finally:
                self.testing_RMSE_metric.to(self.curr_device)
                results = self.model.generate(data, labels=target)
                self.testing_RMSE_metric.update(results[0], results[1])
