# -*- coding: utf-8 -*-
import logging
import os

import click
import torch

from src.data.irmas import (load_irmas_dataset, load_irmas_episodes,
                            prepare_irmas_dataloaders)
from src.types.config import DatasetConfig
from src.types.dataset import DatasetType
from src.utils.yaml_utils import load_yaml_data


def handle_irmas_dataset(
        config: DatasetConfig,
        input_filepath: str,
        output_filepath: str):
    logger = logging.getLogger(__name__)

    train_data, val_data = load_irmas_dataset(
        input_filepath, sample_rate=config.sample_rate)
    logger.info(
        f"Loaded irmas data. Train_tracks: {len(train_data.tracks)}, val_tracks: {len(val_data.tracks)}.")

    train_episodes, val_episodes = load_irmas_episodes(train_data, val_data, n_way=config.n_way,
                                                       n_support=config.n_support,
                                                       n_query=config.n_query, n_train_episodes=config.n_train_episodes,
                                                       n_val_episodes=config.n_val_episodes)
    logger.info("Loaded irmas episodes.")

    train_loader, val_loader = prepare_irmas_dataloaders(
        train_episodes, val_episodes, num_workers=config.num_workers)
    logger.info("Loaded irmas loaders.")

    os.makedirs(os.path.join(output_filepath, "irmas"), exist_ok=True)
    torch.save(train_loader, os.path.join(
        output_filepath, "irmas", "train_loader.pt"))
    torch.save(val_loader, os.path.join(
        output_filepath, "irmas", "val_loader.pt"))
    logger.info(
        f"Saved dataloaders to: {os.path.join(output_filepath, 'irmas')}")


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
