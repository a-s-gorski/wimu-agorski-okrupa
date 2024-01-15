# -*- coding: utf-8 -*-
import logging
import os

import click
import torch

from src.data.irmas import (IRMAS, load_irmas_dataset)
from src.data.tinysol import TinySOL
from src.data.good_sounds import GoodSounds
from src.data.medley_solos_db import MedleySolosDb
from src.data.dataset_handler import DatasetHandler
from src.types.config import DatasetConfig
from src.types.dataset import DatasetType
from src.types.model import ModelEpisodeType
from src.utils.yaml_utils import load_yaml_data

def get_dataset_loader(dataset_name: str, 
                        dataset_handler: DatasetHandler, 
                        config: DatasetConfig,
                        output_filepath: str,
                        train_data, val_data, logger):
    train_episodes, val_episodes = dataset_handler.load_episodes(train_data, val_data, config.n_way, config.n_support, config.n_distractor, config.n_query, config.n_train_episodes, config.n_val_episodes, ModelEpisodeType[config.model_type].value)
    logger.info(f"Loaded {dataset_name} episodes.")

    train_loader, val_loader = dataset_handler.prepare_dataloaders(train_episodes, val_episodes, config.num_workers)
    logger.info(f"Loaded {dataset_name} loaders.")

    os.makedirs(os.path.join(output_filepath, dataset_name), exist_ok=True)
    torch.save(train_loader, os.path.join(
        output_filepath, dataset_name, "train_loader.pt"))
    torch.save(val_loader, os.path.join(
        output_filepath, dataset_name, "val_loader.pt"))
    logger.info(
        f"Saved dataloaders to: {os.path.join(output_filepath, dataset_name)}")


def handle_irmas_dataset(
        config: DatasetConfig,
        input_filepath: str,
        output_filepath: str):
    logger = logging.getLogger(__name__)

    dataset_handler = DatasetHandler(IRMAS)
    train_data, val_data = load_irmas_dataset(
        input_filepath, sample_rate=config.sample_rate)
    logger.info(
        f"Loaded irmas data. Train_tracks: {len(train_data.tracks)}, val_tracks: {len(val_data.tracks)}.")
    get_dataset_loader('irmas', dataset_handler, config, output_filepath, train_data, val_data, logger)

def handle_tiny_sol_dataset(
        config: DatasetConfig,
        input_filepath: str,
        output_filepath: str):
    logger = logging.getLogger(__name__)

    dataset_handler = DatasetHandler(TinySOL)
    train_data, val_data = dataset_handler.load_dataset(
        input_filepath, sample_rate=config.sample_rate)
    logger.info(
        f"Loaded tiny_sol data. Train_tracks: {len(train_data.tracks)}, val_tracks: {len(val_data.tracks)}.")
    get_dataset_loader('tiny_sol', dataset_handler, config, output_filepath, train_data, val_data, logger)

def handle_good_sounds_dataset(
        config: DatasetConfig,
        input_filepath: str,
        output_filepath: str):
    logger = logging.getLogger(__name__)

    dataset_handler = DatasetHandler(GoodSounds)
    train_data, val_data = dataset_handler.load_dataset(
        input_filepath, sample_rate=config.sample_rate)
    logger.info(
        f"Loaded good_sounds data. Train_tracks: {len(train_data.tracks)}, val_tracks: {len(val_data.tracks)}.")
    get_dataset_loader('good_sounds', dataset_handler, config, output_filepath, train_data, val_data, logger)

def handle_medley_solos_db_dataset(
        config: DatasetConfig,
        input_filepath: str,
        output_filepath: str):
    logger = logging.getLogger(__name__)

    dataset_handler = DatasetHandler(MedleySolosDb)
    train_data, val_data = dataset_handler.load_dataset(
        input_filepath, sample_rate=config.sample_rate)
    logger.info(
        f"Loaded medley_solos_db data. Train_tracks: {len(train_data.tracks)}, val_tracks: {len(val_data.tracks)}.")
    get_dataset_loader('medley_solos_db', dataset_handler, config, output_filepath, train_data, val_data, logger)


@click.command()
@click.argument('dataset_type', type=DatasetType)
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(dataset_type, config_path, input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Started loading the dataset.')

    yaml_data = load_yaml_data(config_path)
    dataset_config = DatasetConfig(**yaml_data)

    logger.info(f"Loaded dataset config with values: {str(dataset_config)}")

    match dataset_type:
        case DatasetType.IRMAS:
            handle_irmas_dataset(
                dataset_config, input_filepath, output_filepath)
        case DatasetType.GOOD_SOUNDS:
            handle_good_sounds_dataset(
                dataset_config, input_filepath, output_filepath)
        case DatasetType.WAVEFILES:
            # TODO - implement wavefiles
            pass
        case DatasetType.TINY_SOL:
            handle_tiny_sol_dataset(
                dataset_config, input_filepath, output_filepath)
        case DatasetType.MEDLEY_SOLOS_DB:
            handle_medley_solos_db_dataset(
                dataset_config, input_filepath, output_filepath)
        case _:
            pass


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
