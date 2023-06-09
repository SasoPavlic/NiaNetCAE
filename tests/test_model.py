import argparse
import random
import unittest
import uuid
from datetime import datetime
from pathlib import Path

import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from nianetcae.dataloaders.nyu_dataloader import NYUDataset
from nianetcae.experiments.dnn_ae_experiment import DNNAEExperiment, FineTuneLearningRateFinder
from nianetcae.models.conv_ae import ConvAutoencoder
from nianetcae.niapy_extension.wrapper import *
from nianetcae.storage.database import SQLiteConnector


class TestModel(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        RUN_UUID = uuid.uuid4().hex
        print(f"RUN UUID: {RUN_UUID}")

        torch.set_float32_matmul_precision("medium")
        parser = argparse.ArgumentParser(description='Generic runner for Convolutional AE models')
        parser.add_argument('--config', '-c',
                            dest="filename",
                            metavar='FILE',
                            help='path to the config file',
                            default='configs/main_config.yaml')

        args = parser.parse_args()
        with open(args.filename, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        config['logging_params']['save_dir'] += RUN_UUID + '/'
        Path(config['logging_params']['save_dir']).mkdir(parents=True, exist_ok=True)

        conn = SQLiteConnector(config['logging_params']['db_storage'], f"solutions")  # _{RUN_UUID}")
        seed_everything(config['exp_params']['manual_seed'], True)

        datamodule = NYUDataset(**config["data_params"])
        datamodule.setup()

        self.RUN_UUID = RUN_UUID
        self.config = config
        self.conn = conn
        self.datamodule = datamodule
        self.iteration = random.random() * 100

    def generate_solution(self):
        # Create an array with 5 elements of numbers between 0,1
        solution = [random.random() for _ in range(4)]
        print(f"Solution: {solution}")
        return solution

    def test_architecture_trainees(self):

        try:
            trainees = False
            for x in range(3):

                solution = self.generate_solution()

                model = ConvAutoencoder(solution, **self.config)
                existing_entry = self.conn.get_entries(hash_id=model.hash_id)
                path = self.config['logging_params']['save_dir'] + str(self.iteration) + "_TEST_" + model.hash_id
                Path(path).mkdir(parents=True, exist_ok=True)

                if existing_entry.shape[0] > 0:
                    fitness = existing_entry['fitness'][0]
                    print(f"Model for this solution already exists")
                    trainees = True

                else:
                    """Punishing bad decisions"""
                    if len(model.encoding_layers) == 0 or len(model.decoding_layers) == 0:
                        CADL = int(9e10)
                        trainees = False
                    else:
                        experiment = DNNAEExperiment(model, self.config['exp_params'],
                                                     self.config['data_params']['horizontal_dim'])
                        tb_logger = TensorBoardLogger(save_dir=self.config['logging_params']['save_dir'],
                                                      name=str(self.iteration) + "_TEST_" + model.hash_id)

                        trainer = Trainer(logger=tb_logger,
                                          enable_progress_bar=True,
                                          accelerator="cuda",
                                          devices=1,
                                          default_root_dir=tb_logger.root_dir,
                                          log_every_n_steps=32,
                                          # auto_select_gpus=True,

                                          callbacks=[
                                              LearningRateMonitor(),
                                              # BatchSizeFinder(mode="power", steps_per_trial=3),
                                              # FineTuneLearningRateFinder(update_on_x_epoch=10, **config['lr_finder']),
                                              # EarlyStopping(**config['early_stop'],
                                              #               verbose=False,
                                              #               check_finite=True),
                                              ModelCheckpoint(save_top_k=1,
                                                              dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                                              monitor="loss",
                                                              save_last=True)
                                          ],
                                          # strategy=DDPPlugin(find_unused_parameters=False),
                                          **self.config['trainer_params'])

                        print(f"======= Training {self.config['model_params']['name']} =======")
                        print(f'\nTraining start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
                        trainer.fit(experiment, datamodule=self.datamodule)
                        print(f'\nTraining end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
                        trainer.test(experiment, datamodule=self.datamodule)
                        trainees = True

        except Exception as e:
            trainees = False
            print(f"Exception: {e}")
            self.assertEqual(True, trainees)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestModel('test_architecture_trainees'))
    runner = unittest.TextTestRunner()
