import torch
import torchmetrics
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import LearningRateFinder
from torch import Tensor
from log import Log
from nianetcae.experiments.metrics import ConvAutoencoderDepthLoss, RootMeanAbsoluteError, AbsoluteRelativeDifference, \
    Log10Metric, \
    Delta1, Delta2, Delta3, Metrics
from nianetcae.models.conv_ae import ConvAutoencoder


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs['lr_finder'])
        self.tune_n_epochs = kwargs['tune_n_epochs']
        self.previous_loss = float('inf')

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.tune_n_epochs == 0 or trainer.current_epoch == 0:
            if pl_module.train_loss is not None:
                loss = pl_module.train_loss['loss'].item()
                if loss < self.previous_loss:
                    Log.debug(f"\nLoss decreased from {self.previous_loss} to {loss}")

                elif loss > self.previous_loss:
                    Log.debug(f"\nLoss increased from {self.previous_loss} to {loss}")
                    self.lr_find(trainer, pl_module)

                self.previous_loss = pl_module.train_loss['loss'].item()

            else:
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
        self.train_loss = None
        self.val_loss = None

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

    def get_metrics(self):

        return Metrics(self.MSE_score.item(),
                       self.RMSE_score.item(),
                       self.MAE_score.item(),
                       self.ABS_REL_score.item(),
                       self.LOG10_score.item(),
                       self.DELTA1_score.item(),
                       self.DELTA2_score.item(),
                       self.DELTA3_score.item(),
                       self.CADL_score.item())

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
        self.train_loss = self.model.loss_function(self.curr_device,
                                                   **results,
                                                   optimizer_idx=optimizer_idx,
                                                   batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in self.train_loss.items()}, prog_bar=True, sync_dist=True,
                      on_step=False,
                      on_epoch=True, batch_size=batch['image'].shape[0])

        torch.cuda.empty_cache()
        return self.train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        torch.cuda.empty_cache()
        results = self.forward(batch)
        self.curr_device = batch['image'].device
        self.val_loss = self.model.loss_function(self.curr_device,
                                                 **results,
                                                 optimizer_idx=optimizer_idx,
                                                 batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in self.val_loss.items()}, prog_bar=True, sync_dist=True,
                      on_step=False,
                      on_epoch=True, batch_size=batch['image'].shape[0])

        torch.cuda.empty_cache()
        return self.val_loss['loss']

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

        self.log_dict(dict({'MSE': self.MSE_score, #Low is better
                            'RMSE': self.RMSE_score,#Low is better
                            'MAE': self.MAE_score, #Low is better
                            'ABS_REL': self.ABS_REL_score, #Low is better
                            'LOG10': self.LOG10_score, #Low is better
                            'DELTA1': self.DELTA1_score, #High is better
                            'DELTA2': self.DELTA2_score,#High is better
                            'DELTA3': self.DELTA3_score,#High is better
                            'CADL': self.CADL_score}), #Low is better
                      prog_bar=True, sync_dist=True, on_step=False,
                      on_epoch=True, batch_size=batch['image'].shape[0])

        torch.cuda.empty_cache()
