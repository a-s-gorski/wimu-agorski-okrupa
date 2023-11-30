import os
import sys
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import music_fsl.util as util
import torch
from torch.utils.data import DataLoader

from src.data.class_conditional_dataset import ClassConditionalDataset
from src.data.episode_dataset import EpisodeDataset

class MedleySolosDb(ClassConditionalDataset):
    """
    Initialize a `Medley-solos-DB Dataset Loader` dataset instance.
    
     Each of these clips contains a single instrument among a taxonomy of eight:

        0. clarinet,
        1. distorted electric guitar,
        2. female singer,
        3. flute,
        4. piano,
        5. tenor saxophone,
        6. trumpet, and
        7. violin.

    Args:
        instruments (List[str]): A list of instruments to include in the dataset.
        duration (float): The duration of each audio clip in the dataset (in seconds).
        sample_rate (int): The sample rate of the audio clips in the dataset (in Hz).
    """

    INSTRUMENTS = [
        'clarinet',
        'distorted electric guitar',
        'female singer',
        'flute',
        'piano',
        'tenor saxophone',
        'trumpet',
        'violin'
    ]

    def __init__(self, 
            instruments: List[str] = None,
            duration: float = 1.0, 
            sample_rate: int = 16000,
            dataset_path: str = None,
        ):
        if instruments is None:
            instruments = self.INSTRUMENTS

        self.instruments = instruments  
        self.duration = duration
        self.sample_rate = sample_rate

        metadata = os.path.join(dataset_path, 'Medley-solos-DB_metadata.csv')
        df = pd.read_csv(metadata) 

        # make sure the instruments passed in are valid
        for instrument in instruments:
            assert instrument in self.INSTRUMENTS, f"{instrument} is not a valid instrument"

        # load all tracks for this instrument
        self.tracks = []
        for index, row in df.iterrows():
            if row['instrument'] in self.instruments:
                file_name = f'Medley-solos-DB_{row["subset"]}-{row["instrument_id"]}_{row["uuid4"]}.wav.wav'
                wav_file = os.path.join(dataset_path, 'Medley-solos-DB', file_name)
                self.tracks.append([wav_file, row['instrument']])

    @property
    def classlist(self) -> List[str]:
        return self.instruments

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        # cache it in self._class_to_indices 
        # so we don't have to recompute it every time
        if not hasattr(self, "_class_to_indices"):
            self._class_to_indices = defaultdict(list)
            for i, track in enumerate(self.tracks):
                self._class_to_indices[track[1]].append(i)

        return self._class_to_indices

    def __getitem__(self, index) -> Dict:
        # load the track for this index
        track = self.tracks[index]

        # load the excerpt
        data = util.load_excerpt(track[0], self.duration, self.sample_rate)
        data["label"] = track[1]

        return data

    def __len__(self) -> int:
        return len(self.tracks)


def load_madleysolosdb_dataset(
        path: str, sample_rate: Optional[int] = 16000) -> Tuple[IRMAS, IRMAS]:
    TRAIN_INSTRUMENTS = [
            'flute', 'piano', 'tenor saxophone', 'trumpet', 'violin'
        ]

    TEST_INSTRUMENTS = [
            'clarinet', 'distorted electric guitar', 'female singer',
        ]
    train_data = MedleySolosDb(
        instruments=TRAIN_INSTRUMENTS, 
        sample_rate=sample_rate,
        dataset_path = path
    )

    val_data = MedleySolosDb(
        instruments=TEST_INSTRUMENTS, 
        sample_rate=sample_rate,
        dataset_path = path
    )
    return train_data, val_data


def load_madleysolosdb_episodes(train_data: MedleySolosDb,
                        val_data: MedleySolosDb,
                        n_way: int = 3,
                        n_support: int = 5,
                        n_query: int = 20,
                        n_train_episodes=1000,
                        n_val_episodes=50) -> Tuple[EpisodeDataset,
                                                    EpisodeDataset]:
    train_episodes = EpisodeDataset(
        dataset=train_data,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
        n_episodes=n_train_episodes
    )

    val_episodes = EpisodeDataset(
        dataset=val_data,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
        n_episodes=n_val_episodes
    )

    return train_episodes, val_episodes


def prepare_madleysolosdb_dataloaders(train_episodes: EpisodeDataset,
                              val_episodes: EpisodeDataset,
                              num_workers: int = 12) -> Tuple[DataLoader,
                                                              DataLoader]:
    train_loader = DataLoader(
        train_episodes, batch_size=None, num_workers=num_workers)
    val_loader = DataLoader(
        val_episodes, batch_size=None, num_workers=num_workers)

    return train_loader, val_loader


def load_madleysolosdb_dataloaders(input_path: str) -> Tuple[DataLoader, DataLoader]:
    train_loader = torch.load(os.path.join(input_path, "train_loader.pt"))
    val_loader = torch.load(os.path.join(input_path, "val_loader.pt"))

    return train_loader, val_loader
