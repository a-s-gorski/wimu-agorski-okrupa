from typing import Optional

import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from lightning.pytorch.profilers import SimpleProfiler

from src.types.config import TrainingConfig


def get_profiler(config: TrainingConfig) -> Optional[SimpleProfiler]:
    match config.profiler:
        case "simple":
            return SimpleProfiler(filename="profile.txt")
        case _:
            return None


def get_logger(config: TrainingConfig) -> Optional[TensorBoardLogger]:
    match config.logger:
        case "tensorboard":
            return TensorBoardLogger(save_dir=config.logger_output_dir, name="logs")
        case "wandb":
            return WandbLogger(save_dir=config.logger_output_dir, log_model='all')
        case "csv":
            return CSVLogger(save_dir=config.logger_output_dir)
        case _:
            return None


def get_trainer(config: TrainingConfig, checkpoint_path: str) -> pl.Trainer:
    logger = get_logger(config)
    profiler = get_profiler(config)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        profiler=profiler,
        logger=logger,
        default_root_dir=checkpoint_path,
    )

    return trainer
