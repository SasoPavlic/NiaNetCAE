from pathlib import Path

import torch

from niapy.algorithms.basic import ParticleSwarmAlgorithm, DifferentialEvolution, FireflyAlgorithm, GeneticAlgorithm
from niapy.algorithms.modified import SelfAdaptiveDifferentialEvolution

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import Trainer
from tabulate import tabulate

from nianetcae.experiments.dnn_ae_experiment import DNNAEExperiment
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
        print("=================================================================================================")
        print(f"ITERATION: {self.iteration}")
        print(f"SOLUTION : {solution}")
        self.iteration += 1

        model = ConvAutoencoder(solution, **config)
        existing_entry = conn.get_entries(hash_id=model.hash_id)
        path = config['logging_params']['save_dir'] + str(self.iteration) + "_" + alg_name + "_" + model.hash_id
        Path(path).mkdir(parents=True, exist_ok=True)

        if existing_entry.shape[0] > 0:
            fitness = existing_entry['fitness'][0]
            print(f"Model for this solution already exists")
            return fitness

        else:
            # TODO Find a more optimal way
            """Punishing bad decisions"""
            if len(model.encoding_layers) == 0 or len(model.decoding_layers) == 0:
                CADL = int(9e10)
            else:
                experiment = DNNAEExperiment(model, config['exp_params'], config['model_params']['tensor_dim'])
                config['trainer_params']['max_epochs'] = model.num_epochs
                tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                              name=str(self.iteration) + "_" + alg_name + "_" + model.hash_id)

                runner = Trainer(logger=tb_logger,
                                 enable_progress_bar=True,
                                 accelerator="cuda",
                                 devices=1,
                                 # auto_select_gpus=True,

                                 callbacks=[
                                     LearningRateMonitor(),
                                     # BatchSizeFinder(),
                                     # LearningRateFinder(attr_name="lr"),
                                     ModelCheckpoint(save_top_k=1,
                                                     dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                                     monitor="loss",
                                                     save_last=True)
                                 ],
                                 # strategy=DDPPlugin(find_unused_parameters=False),
                                 **config['trainer_params'])

                print(f"======= Training {config['model_params']['name']} =======")
                print(f'\nTraining start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
                runner.fit(experiment, datamodule=datamodule)
                print(f'\nTraining end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
                runner.test(experiment, datamodule=datamodule)
                CADL = experiment.CADL_score.item()

            complexity = (model.num_epochs ** 2) + (model.num_layers * 100) + (model.bottleneck_size * 10)
            fitness = (CADL * 1000) + (complexity / 100)

            print(tabulate([[CADL, complexity, fitness]], headers=["RMSE", "AUC", "Complexity", "Fitness"],
                           tablefmt="pretty"))
            conn.post_entries(model, fitness, solution, CADL, complexity, alg_name, self.iteration)
            torch.save(model.state_dict(), path + f"/model.pt")

            return fitness


def solve_architecture_problem():
    print(f'Program start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
    print(f"RUN UUID: {RUN_UUID}")
    """
    Dimensionality:
    y1: topology shape,
    y2: number of neurons per layer,
    y3: number of layers,
    y4: activation function
    y5: number of epochs,
    y6: learning rate
    y7: optimizer algorithm.
    """
    DIMENSIONALITY = 7

    runner = ExtendedRunner(
        config['logging_params']['save_dir'],
        dimension=DIMENSIONALITY,
        max_evals=1000,
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

    print("=====================================SEARCH STARTED==============================================")
    final_solutions = runner.run(export='json', verbose=True)
    print("=====================================SEARCH COMPLETED============================================")

    print(f"Solutions: {final_solutions}")
    best_solution, best_algorithm = conn.best_results()
    best_model = ConvAutoencoder(best_solution, **config)
    model_file = config['logging_params']['save_dir'] + f"{best_algorithm}_{best_model.hash_id}.pt"
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    torch.save(best_model.state_dict(), model_file)
    print(f"Best model saved to: {model_file}")
    print(f'\n Program end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
