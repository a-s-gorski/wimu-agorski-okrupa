"""
`train.py` - Script for training models on different datasets.

This script provides a command-line interface for training models on various datasets.
It currently supports datasets such as IRMAS (Instrument Recognition in Musical Audio) and others.

Usage:
    python train.py <dataset_type> <config_path> <input_filepath> <model_output_filepath>

Arguments:
    dataset_type (str): The type of dataset to train on. Supported values: 'irmas', 'good_sounds', 'wavefiles',
    'tiny_sol'.
    config_path (str): Path to the YAML configuration file containing training parameters.
    input_filepath (str): Path to the input data directory.
    model_output_filepath (str): Path to the output directory for saving the trained models.

Example:
    python train.py irmas path/to/config.yaml path/to/dataset irmas_models/

Note:
    This script uses a YAML configuration file to set up training parameters.
    Ensure that the specified dataset type matches the available cases in the script.
"""
import logging
import os

import click

from src.data.irmas import load_irmas_dataloaders
from src.models.few_shot_learner import FewShotLearner
from src.models.models import get_protypical_net
from src.models.trainer import get_trainer
from src.types.config import TrainingConfig
from src.types.dataset import DatasetType
from src.utils.yaml_utils import load_yaml_data


def handle_irmas_training(
        config: TrainingConfig,
        input_filepath: str,
        model_output_filepath: str):
    logger = logging.getLogger(__name__)

    train_loader, val_loader = load_irmas_dataloaders(
        os.path.join(input_filepath, "irmas"))
    logger.info("Loaded irmas dataloaders.")

    trainer = get_trainer(config=config, checkpoint_path=model_output_filepath)
    logger.info("Created trainer object. Staring training.")

    protype_net = get_protypical_net(config=config)
    model = FewShotLearner(protonet=protype_net)

    trainer.fit(model, train_loader, val_dataloaders=val_loader)


@click.command()
@click.argument('dataset_type', type=DatasetType)
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_output_filepath', type=click.Path())
def main(
        dataset_type: DatasetType,
        config_path,
        input_filepath,
        model_output_filepath):

    logger = logging.getLogger(__name__)
    logger.info(f"Started training the dataset {dataset_type.value}.")

    yaml_data = load_yaml_data(config_path)
    training_config = TrainingConfig(**yaml_data)
    logger.info(f"Loaded training config with params: {str(training_config)}")

    match dataset_type:
        case DatasetType.IRMAS:
            handle_irmas_training(
                training_config, input_filepath, model_output_filepath)
        case DatasetType.GOOD_SOUNDS:
            # TODO - implement good_sounds
            pass
        case DatasetType.WAVEFILES:
            # TODO - implement wavefiles
            pass
        case DatasetType.TINY_SOL:
            # TODO - implement tiny_sol
            pass
        case _:
            pass


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
