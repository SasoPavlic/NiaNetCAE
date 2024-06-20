import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import LearningRateFinder
from torch import Tensor

from log import Log
from nianetcae.experiments.evaluationmetrics import EvaluationMetrics
from nianetcae.models.conv_ae import ConvAutoencoder
from nianetcae.visualize.batch_to_image import visualise_batch


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
    def __init__(self, conv_autoencoder: ConvAutoencoder, **kwargs) -> None:
        super(DNNAEExperiment, self).__init__()

        # https://github.com/Lightning-AI/lightning/issues/4390#issuecomment-717447779
        # self.save_hyperparameters(logger=False)

        self.results = None
        self.model = conv_autoencoder
        self.model_path = kwargs['logging_params']['model_path']
        self.learning_rate = 0.0
        self.params = kwargs['exp_params']
        self.tensor_dim = kwargs['data_params']['horizontal_dim']
        self.curr_device = None
        self.hold_graph = False
        self.train_loss = None
        self.val_loss = None
        self.metrics = EvaluationMetrics()

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

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        results = self.forward(batch)
        self.curr_device = batch['image'].device
        self.train_loss = self.model.loss_function(self.curr_device,
                                                   **results,
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
        results = self.forward(batch)

        self.metrics.to(self.curr_device)

        test_loss = self.model.loss_function(self.curr_device,
                                             **results,
                                             optimizer_idx=optimizer_idx,
                                             batch_idx=batch_idx)

        self.metrics.update(results['output'], results['depth'])
        self.metrics.update_CADL(test_loss['loss'])
        visualise_batch(self.model_path, batch_idx, **results)

        self.results = self.metrics.compute()

        self.log_dict(self.results,
                      prog_bar=True, sync_dist=True, on_step=False,
                      on_epoch=True, batch_size=batch['image'].shape[0])

        torch.cuda.empty_cache()
