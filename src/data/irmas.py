import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import music_fsl.util as util
import torch
from torch.utils.data import DataLoader

from src.data.class_conditional_dataset import ClassConditionalDataset
from src.data.episode_dataset import EpisodeDataset


class IRMAS(ClassConditionalDataset):
    """
    Initialize a `IRMAS` dataset instance.

    Args:
        instruments (List[str]): A list of instruments to include in the dataset.
        duration (float): The duration of each audio clip in the dataset (in seconds).
        sample_rate (int): The sample rate of the audio clips in the dataset (in Hz).
    """

    INSTRUMENTS = [
        'Cello', 'Clarinet', 'Flute', 'Acoustic guitar',
        'Electric guitar', 'Organ', 'Piano',
        'Saxophone', 'Trumpet', 'Violin', 'Human singing voice'
    ]

    INSTRUMENTS_KEY = {
        'cel': 'Cello',
        'cla': 'Clarinet',
        'flu': 'Flute',
        'gac': 'Acoustic guitar',
        'gel': 'Electric guitar',
        'org': 'Organ',
        'pia': 'Piano',
        'sax': 'Saxophone',
        'tru': 'Trumpet',
        'vio': 'Violin',
        'voi': 'Human singing voice'}

    def __init__(self,
                 instruments: List[str] = None,
                 instruments_key: Dict[str, str] = None,
                 duration: float = 1.0,
                 sample_rate: int = 16000,
                 dataset_path: str = 'irmas',
                 val=False,
                 ):
        if instruments_key is None:
            instruments_key = self.INSTRUMENTS_KEY

        if instruments is None:
            instruments = self.INSTRUMENTS

        if val is False and dataset_path == 'irmas':
            dir_path = os.path.dirname(os.path.realpath(__file__))
            dataset_path = os.path.join(
                os.path.dirname(dir_path), dataset_path)

        self.instruments = instruments
        self.instruments_key = instruments_key
        self.duration = duration
        self.sample_rate = sample_rate

        # initialize IRMAS path
        if not val:
            if os.path.exists(dataset_path):
                self.dataset_path = dataset_path
            else:
                sys.exit("Dataset path does not exist")

        if val:
            self.dataset_path = dataset_path

        # make sure the instruments passed in are valid
        for instrument in instruments:
            assert instrument in self.INSTRUMENTS, f"{instrument} is not a valid instrument"

        # load all tracks for this instrument
        if val:
            self.load_all_tracks_val()
        else:
            self.load_all_tracks()

    def load_all_tracks(self):
        self.tracks = []
        for subdir, dirs, files in os.walk(self.dataset_path):
            for dir in dirs:
                if dir in self.instruments_key.keys():
                    for subdir_dir, dirs_dir, files_dir in os.walk(
                            os.path.join(subdir, dir)):
                        for file in files_dir:
                            self.tracks.append(
                                [os.path.join(subdir, dir, file), self.instruments_key[dir]])

    def load_all_tracks_val(self):
        self.tracks = []
        for paths in self.dataset_path:
            for subdir, dirs, files in os.walk(paths, topdown=True):
                for file in sorted(files)[::2]:
                    f = open(os.path.join(subdir, file), "rb")
                    tags = f.readlines()
                    for tag in tags:
                        tag = tag[:3].decode("utf-8")
                        f.close()
                        if tag in self.instruments_key.keys():
                            wav_file = os.path.join(
                                subdir, file).replace('.txt', '.wav')
                            if os.path.exists(wav_file):
                                self.tracks.append(
                                    [wav_file, self.instruments_key[tag]])

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


def load_irmas_dataset(
        path: str, sample_rate: Optional[int] = 16000) -> Tuple[IRMAS, IRMAS]:
    TRAIN_INSTRUMENTS = [
        'Flute', 'Organ', 'Saxophone',
        'Trumpet', 'Violin', 'Electric guitar'
    ]

    TRAIN_INSTRUMENTS_KEY = {
        'flu': 'Flute',
        'org': 'Organ',
        'sax': 'Saxophone',
        'tru': 'Trumpet',
        'vio': 'Violin',
        'gel': 'Electric guitar'}

    TEST_INSTRUMENTS = [
        'Cello', 'Piano', 'Clarinet', 'Acoustic guitar', 'Human singing voice'
    ]

    TEST_INSTRUMENTS_KEY = {
        'cel': 'Cello',
        'gac': 'Acoustic guitar',
        'pia': 'Piano',
        'voi': 'Human singing voice',
        'cla': 'Clarinet'}
    train_data = IRMAS(
        instruments=TRAIN_INSTRUMENTS,
        instruments_key=TRAIN_INSTRUMENTS_KEY,
        sample_rate=sample_rate,
        dataset_path=path
    )

    val_data = IRMAS(
        instruments=TEST_INSTRUMENTS,
        instruments_key=TEST_INSTRUMENTS_KEY,
        sample_rate=sample_rate,
        dataset_path=path
    )
    return train_data, val_data


def load_irmas_episodes(train_data: IRMAS,
                        val_data: IRMAS,
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


def prepare_irmas_dataloaders(train_episodes: EpisodeDataset,
                              val_episodes: EpisodeDataset,
                              num_workers: int = 12) -> Tuple[DataLoader,
                                                              DataLoader]:
    train_loader = DataLoader(
        train_episodes, batch_size=None, num_workers=num_workers)
    val_loader = DataLoader(
        val_episodes, batch_size=None, num_workers=num_workers)

    return train_loader, val_loader


def load_irmas_dataloaders(input_path: str) -> Tuple[DataLoader, DataLoader]:
    train_loader = torch.load(os.path.join(input_path, "train_loader.pt"))
    val_loader = torch.load(os.path.join(input_path, "val_loader.pt"))

    return train_loader, val_loader
