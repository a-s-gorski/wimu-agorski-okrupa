import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.episode_dataset import EpisodeDataset, EpisodeDatasetUnlabeled

class DatasetHandler():
    def __init__(self, Dataset):
        self.Dataset = Dataset
        self.TRAIN_INSTRUMENTS = Dataset.TRAIN_INSTRUMENTS
        self.TEST_INSTRUMENTS = Dataset.TEST_INSTRUMENTS


    def load_dataset(self, path: str, sample_rate: Optional[int] = 16000):

        train_data = self.Dataset(
            instruments=self.Dataset.TRAIN_INSTRUMENTS,
            sample_rate=sample_rate,
            dataset_path=path
        )

        val_data = self.Dataset(
            instruments=self.Dataset.TEST_INSTRUMENTS,
            sample_rate=sample_rate,
            dataset_path=path
        )
        return train_data, val_data
    

    def load_episodes(self, train_data, val_data, n_way: int = 5, n_support: int = 5, n_query: int = 20, n_train_episodes: int = 1000, n_val_episodes: int = 50, episode_type = EpisodeDataset) -> Tuple[EpisodeDataset, EpisodeDataset]:
        train_episodes = episode_type(
            dataset=train_data,
            n_way=n_way,
            n_support=n_support,
            n_query=n_query,
            n_episodes=n_train_episodes
        )

        val_episodes = episode_type(
            dataset=val_data,
            n_way=n_way,
            n_support=n_support,
            n_query=n_query,
            n_episodes=n_val_episodes
        )

        return train_episodes, val_episodes


    def prepare_dataloaders(self, train_episodes: EpisodeDataset,
                                val_episodes: EpisodeDataset,
                                num_workers: int = 12) -> Tuple[DataLoader,
                                                                DataLoader]:
        train_loader = DataLoader(
            train_episodes, batch_size=None, num_workers=num_workers)
        val_loader = DataLoader(
            val_episodes, batch_size=None, num_workers=num_workers)

        return train_loader, val_loader


    def load_dataloaders(input_path: str) -> Tuple[DataLoader, DataLoader]:
        train_loader = torch.load(os.path.join(input_path, "train_loader.pt"))
        val_loader = torch.load(os.path.join(input_path, "val_loader.pt"))

        return train_loader, val_loader
