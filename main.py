import uuid
from datetime import datetime

import torch
import argparse
import yaml
from lightning.pytorch import seed_everything

import nianetcae
from nianetcae.storage.database import SQLiteConnector
from pathlib import Path
from nianetcae.dataloaders.nyu_dataloader import NYUDataset

from nianetcae.cae_run import solve_architecture_problem

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
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    config['logging_params']['save_dir'] += RUN_UUID + '/'
    Path(config['logging_params']['save_dir']).mkdir(parents=True, exist_ok=True)

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
