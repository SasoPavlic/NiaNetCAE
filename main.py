import argparse
import uuid
from datetime import datetime
from os import path
from pathlib import Path
import logging.config
from log import Log

import torch
import yaml
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping

import nianetcae
from nianetcae.cae_run import solve_architecture_problem
from nianetcae.dataloaders.nyu_dataloader import NYUDataset
from nianetcae.storage.database import SQLiteConnector

if __name__ == '__main__':

    RUN_UUID = uuid.uuid4().hex
    print(f'Program start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
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
            config = yaml.load(file, Loader=yaml.Loader)  # yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Error while loading config file")
            print(exc)

    config['logging_params']['save_dir'] += RUN_UUID + '/'
    Path(config['logging_params']['save_dir']).mkdir(parents=True, exist_ok=True)

    Log.enable(config['logging_params'])
    Log.header("NiaNetCAE settings")
    Log.info(config['model_params'])

    conn = SQLiteConnector(config['logging_params']['db_storage'], f"solutions")  # _{RUN_UUID}")
    seed_everything(config['exp_params']['manual_seed'], True)

    datamodule = NYUDataset(**config["data_params"])
    datamodule.setup()

    nianetcae.cae_run.RUN_UUID = RUN_UUID
    nianetcae.cae_run.config = config
    nianetcae.cae_run.conn = conn
    nianetcae.cae_run.datamodule = datamodule

    solve_architecture_problem()
    print(f'\n Program end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
