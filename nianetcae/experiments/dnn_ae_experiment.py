import torch
import torchmetrics
from lightning.pytorch import LightningModule
from torch import Tensor

from nianetcae.experiments.metrics import ConvAutoencoderDepthLoss, RootMeanAbsoluteError, AbsoluteRelativeDifference, Log10Metric, \
    Delta1, Delta2, Delta3
from nianetcae.models.base import BaseAutoencoder
from nianetcae.models.conv_ae import ConvAutoencoder
from lightning.pytorch.callbacks import LearningRateFinder


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, update_on_x_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_on_x_epoch = update_on_x_epoch

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.update_on_x_epoch == 0 or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


class DNNAEExperiment(LightningModule):
    def __init__(self, conv_autoencoder: ConvAutoencoder, params: dict, tensor_dim: int) -> None:
        super(DNNAEExperiment, self).__init__()

        # https://github.com/Lightning-AI/lightning/issues/4390#issuecomment-717447779
        # self.save_hyperparameters(logger=False)

        self.model = conv_autoencoder
        self.learning_rate = 0.0
        self.params = params
        self.tensor_dim = tensor_dim
        self.curr_device = None
        self.hold_graph = False

        self.MSE_metric = torchmetrics.MeanSquaredError()
        self.RMSE_metric = RootMeanAbsoluteError()
        self.MAE_metric = torchmetrics.MeanAbsoluteError()
        self.ABS_REL_metric = AbsoluteRelativeDifference()
        self.LOG10_metric = Log10Metric()
        self.DELTA1_metric = Delta1()
        self.DELTA2_metric = Delta2()
        self.DELTA3_metric = Delta3()
        self.CADL_metric = ConvAutoencoderDepthLoss()

        self.MSE_score = None
        self.RMSE_score = None
        self.MAE_score = None
        self.ABS_REL_score = None
        self.LOG10_score = None
        self.DELTA1_score = None
        self.CADL_score = None

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def configure_optimizers(self):

        """When AE does not have any layers"""
        if len(list(self.model.parameters())) == 0:
            self.model.optimizer_name = "Empty"
            return None

        if self.model.optimizer_name == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        elif self.model.optimizer_name == "Adagrad":
            return torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)

        elif self.model.optimizer_name == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        elif self.model.optimizer_name == "RAdam":
            return torch.optim.RAdam(self.model.parameters(), lr=self.learning_rate)

        elif self.model.optimizer_name == "ASGD":
            return torch.optim.ASGD(self.model.parameters(), lr=self.learning_rate)

        elif self.model.optimizer_name == "RPROP":
            return torch.optim.Rprop(self.model.parameters(), lr=self.learning_rate)

        else:
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def training_step(self, batch, batch_idx, optimizer_idx=0):
        torch.cuda.empty_cache()
        results = self.forward(batch)
        self.curr_device = batch['image'].device
        train_loss = self.model.loss_function(self.curr_device,
                                              **results,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, prog_bar=True, sync_dist=True,
                      on_step=False,
                      on_epoch=True, batch_size=batch['image'].shape[0])

        torch.cuda.empty_cache()
        return train_loss['loss']

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        torch.cuda.empty_cache()
        dataloader_iterator = iter(self.trainer.datamodule.test_dataloader())

        while True:
            try:
                batch = next(dataloader_iterator)
                batch['image'] = batch['image'].to(self.curr_device)
                batch['depth'] = batch['depth'].to(self.curr_device)
            except StopIteration:
                break
            finally:
                results = self.forward(batch)

                self.CADL_metric.to(self.curr_device)

                self.MSE_metric.to(self.curr_device)
                self.RMSE_metric.to(self.curr_device)
                self.MAE_metric.to(self.curr_device)
                self.ABS_REL_metric.to(self.curr_device)
                self.LOG10_metric.to(self.curr_device)
                self.DELTA1_metric.to(self.curr_device)
                self.DELTA2_metric.to(self.curr_device)
                self.DELTA3_metric.to(self.curr_device)

                self.MSE_metric.update(results['output'], results['depth'])
                self.RMSE_metric.update(results['output'], results['depth'])
                self.MAE_metric.update(results['output'], results['depth'])
                self.ABS_REL_metric.update(results['output'], results['depth'])
                self.LOG10_metric.update(results['output'], results['depth'])
                self.DELTA1_metric.update(results['output'], results['depth'])
                self.DELTA2_metric.update(results['output'], results['depth'])
                self.DELTA3_metric.update(results['output'], results['depth'])

                test_loss = self.model.loss_function(self.curr_device,
                                                     **results,
                                                     optimizer_idx=optimizer_idx,
                                                     batch_idx=batch_idx)

                self.CADL_metric.update(test_loss['loss'])
                # visualise_batch(**results)

        self.MSE_score = self.MSE_metric.compute()
        self.RMSE_score = self.RMSE_metric.compute()
        self.MAE_score = self.MAE_metric.compute()
        self.ABS_REL_score = self.ABS_REL_metric.compute()
        self.LOG10_score = self.LOG10_metric.compute()
        self.DELTA1_score = self.DELTA1_metric.compute()
        self.DELTA2_score = self.DELTA2_metric.compute()
        self.DELTA3_score = self.DELTA3_metric.compute()
        self.CADL_score = self.CADL_metric.compute()

        self.log_dict(dict({'MSE': self.MSE_score,
                            'RMSE': self.RMSE_score,
                            'MAE': self.MAE_score,
                            'ABS_REL': self.ABS_REL_score,
                            'LOG10': self.LOG10_score,
                            'DELTA1': self.DELTA1_score,
                            'DELTA2': self.DELTA2_score,
                            'DELTA3': self.DELTA3_score,
                            'CADL': self.CADL_score}),
                      prog_bar=True, sync_dist=True, on_step=False,
                      on_epoch=True, batch_size=batch['image'].shape[0])

        torch.cuda.empty_cache()