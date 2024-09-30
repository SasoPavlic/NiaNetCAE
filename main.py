import argparse
import uuid
from datetime import datetime
from pathlib import Path

import torch
import yaml
from lightning.pytorch import seed_everything

import nianetcae
from log import Log
from nianetcae.cae_architecture_search import solve_architecture_problem
from nianetcae.dataloaders.nyu_dataloader import NYUDataLoader
from nianetcae.storage.database import SQLiteConnector

def select_dataloader(config):
    dataset_type = config["data_params"].get("dataset_type", "")

    # Define a mapping of dataset types to DataLoader classes
    dataloader_switch = {
        "NYU2": NYUDataLoader,
    }
    # Get the appropriate DataLoader class based on the dataset_type
    DataLoaderClass = dataloader_switch.get(dataset_type)

    if DataLoaderClass is None:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Initialize the DataLoader with the corresponding parameters
    return DataLoaderClass(**config["data_params"])

if __name__ == '__main__':

    RUN_UUID = uuid.uuid4().hex
    torch.set_float32_matmul_precision("medium")
    parser = argparse.ArgumentParser(description='Generic runner for Convolutional AE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/main_config.yaml')

    parser.add_argument('--algorithms', '-alg',
                        dest="algorithms",
                        metavar='list_of_strings',
                        help='NIA algorithms to use')

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        try:
            config = yaml.load(file, Loader=yaml.Loader)  # yaml.safe_load(file)
        except yaml.YAMLError as exc:
            Log.error("Error while loading config file")
            Log.error(exc)

    config['logging_params']['save_dir'] += RUN_UUID + '/'
    Path(config['logging_params']['save_dir']).mkdir(parents=True, exist_ok=True)

    Log.enable(config['logging_params'])
    Log.info(f'Program start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
    Log.info(f"RUN UUID: {RUN_UUID}")
    Log.info(f"PyTorch was compiled with CUDA version: {torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    Log.info(f"Is CUDA available on this system? {'Yes' if cuda_available else 'No'}")
    Log.info(f"PyTorch version: {torch.__version__}")
    Log.header("NiaNetCAE settings")
    Log.info(config)

    conn = SQLiteConnector(config['logging_params']['db_storage'], f"solutions")  # _{RUN_UUID}")
    seed_everything(config['exp_params']['manual_seed'], True)

    datamodule = select_dataloader(config)
    datamodule.setup()

    nianetcae.cae_architecture_search.RUN_UUID = RUN_UUID
    nianetcae.cae_architecture_search.config = config
    nianetcae.cae_architecture_search.conn = conn
    nianetcae.cae_architecture_search.datamodule = datamodule

    algorithms = []
    if args.algorithms is not None:
        args.algorithms = args.algorithms.split(',')
        algorithms = args.algorithms
    else:
        algorithms = config['nia_search']['algorithms']

    solve_architecture_problem(algorithms)
    Log.info(f'\n Program end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
