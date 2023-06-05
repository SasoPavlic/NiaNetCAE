from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, BatchSizeFinder
from lightning.pytorch.loggers import TensorBoardLogger
from niapy.algorithms.basic import ParticleSwarmAlgorithm, DifferentialEvolution, FireflyAlgorithm, GeneticAlgorithm
from niapy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from tabulate import tabulate

from log import Log
from nianetcae.experiments.dnn_ae_experiment import DNNAEExperiment, FineTuneLearningRateFinder
from nianetcae.models.conv_ae import ConvAutoencoder
from nianetcae.niapy_extension.wrapper import *

RUN_UUID = None
config = None
conn = None
datamodule = None


class CONVAEArchitecture(ExtendedProblem):

    def __init__(self, dimension):
        super().__init__(dimension=dimension, lower=0, upper=1)
        self.iteration = 0

    def _evaluate(self, solution, alg_name):
        Log.debug("=================================================================================================")
        Log.debug(f"ITERATION: {self.iteration}")
        Log.debug(f"SOLUTION : {solution}")
        self.iteration += 1

        model = ConvAutoencoder(solution, **config)
        existing_entry = conn.get_entries(hash_id=model.hash_id)
        path = config['logging_params']['save_dir'] + str(self.iteration) + "_" + alg_name + "_" + model.hash_id
        Path(path).mkdir(parents=True, exist_ok=True)

        if existing_entry.shape[0] > 0:
            fitness = existing_entry['fitness'][0]
            Log.info(f"Model for this solution already exists")
            return fitness

        else:
            # TODO Find a more optimal way
            """Punishing bad decisions"""
            if len(model.encoding_layers) == 0 or len(model.decoding_layers) == 0:
                CADL = int(9e10)
            else:
                experiment = DNNAEExperiment(model, config['exp_params'], config['data_params']['horizontal_dim'])
                tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                              name=str(self.iteration) + "_" + alg_name + "_" + model.hash_id)

                trainer = Trainer(logger=tb_logger,
                                  enable_progress_bar=True,
                                  accelerator="cuda",
                                  devices=1,
                                  default_root_dir=tb_logger.root_dir,
                                  log_every_n_steps=50,
                                  # auto_select_gpus=True,

                                  callbacks=[
                                      LearningRateMonitor(),
                                      # BatchSizeFinder(mode="power", steps_per_trial=3),
                                      FineTuneLearningRateFinder(**config['fine_tune_lr_finder']),
                                      # EarlyStopping(**config['early_stop'],
                                      #               verbose=False,
                                      #               check_finite=True),
                                      ModelCheckpoint(save_top_k=1,
                                                      dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                                      monitor="loss",
                                                      save_last=True)
                                  ],
                                  # strategy=DDPPlugin(find_unused_parameters=False),
                                  **config['trainer_params'])

                Log.info(f"======= Training {config['model_params']['name']} =======")
                Log.info(f'\nTraining start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
                trainer.fit(experiment, datamodule=datamodule)
                Log.info(f'\nTraining end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
                trainer.test(experiment, datamodule=datamodule)

                CADL = experiment.CADL_score.item()

            complexity = (model.num_layers * 100) + (model.bottleneck_size * 10)
            fitness = (CADL * 1000) + (complexity / 100)

            Log.debug(tabulate([[CADL, complexity, fitness]], headers=["RMSE", "AUC", "Complexity", "Fitness"],
                           tablefmt="pretty"))
            conn.post_entries(model, fitness, solution, CADL, complexity, alg_name, self.iteration)
            torch.save(model.state_dict(), path + f"/model.pt")

            # TODO Fix when NaN
            if np.isnan(fitness):
                fitness = int(9e10)
            return fitness


def solve_architecture_problem():
    """
    Dimensionality:
    y1: number of neurons per layer,
    y2: number of layers,
    y3: activation function
    y4: optimizer algorithm.
    """
    DIMENSIONALITY = 4

    runner = ExtendedRunner(
        config['logging_params']['save_dir'],
        dimension=DIMENSIONALITY,
        max_evals=100,
        runs=1,
        algorithms=[
            ParticleSwarmAlgorithm(),
            DifferentialEvolution(),
            FireflyAlgorithm(),
            SelfAdaptiveDifferentialEvolution(),
            GeneticAlgorithm()
        ],
        problems=[
            CONVAEArchitecture(DIMENSIONALITY)
        ]
    )

    Log.info("=====================================SEARCH STARTED==============================================")
    final_solutions = runner.run(export='json', verbose=True)
    Log.info("=====================================SEARCH COMPLETED============================================")

    Log.info(f"Solutions: {final_solutions}")
    best_solution, best_algorithm = conn.best_results()
    best_model = ConvAutoencoder(best_solution, **config)
    model_file = config['logging_params']['save_dir'] + f"{best_algorithm}_{best_model.hash_id}.pt"
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    torch.save(best_model.state_dict(), model_file)
    Log.info(f"Best model saved to: {model_file}")
