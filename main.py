import warnings
import uuid
import torch
import argparse
import yaml
from lightning.pytorch import seed_everything

import nianetcae
from nianetcae.storage.database import SQLiteConnector
from pathlib import Path
from nianetcae.dataloaders.images import NYUDataset

from nianetcae.cae_run import solve_architecture_problem

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    torch.set_float32_matmul_precision("medium")

    RUN_UUID = uuid.uuid4().hex
    parser = argparse.ArgumentParser(description='Generic runner for DNN AE models')
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
