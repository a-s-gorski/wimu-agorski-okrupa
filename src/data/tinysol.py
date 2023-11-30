import os
import pandas as pd
from collections import defaultdict
from typing import Dict, List

import music_fsl.util as util

from src.data.class_conditional_dataset import ClassConditionalDataset


class TinySOL(ClassConditionalDataset):
    """
    Initialize a `TinySOL` dataset instance.

    Args:
        instruments (List[str]): A list of instruments to include in the dataset.
        duration (float): The duration of each audio clip in the dataset (in seconds).
        sample_rate (int): The sample rate of the audio clips in the dataset (in Hz).
    """

    INSTRUMENTS = [
        'Bassoon', 'Viola', 'Trumpet in C', 'Bass Tuba',
        'Alto Saxophone', 'French Horn', 'Violin', 
        'Flute', 'Contrabass', 'Trombone', 'Cello', 
        'Clarinet in Bb', 'Oboe', 'Accordion'
    ]

    TRAIN_INSTRUMENTS = [
        'French Horn', 'Violin', 'Flute', 'Contrabass', 'Trombone', 'Cello', 'Clarinet in Bb', 'Oboe', 'Accordion'
    ]

    TEST_INSTRUMENTS = [
        'Bassoon', 'Viola', 'Trumpet in C', 'Bass Tuba', 'Alto Saxophone'
    ]


    def __init__(self, 
                    instruments: List[str] = None,
                    duration: float = 1.0, 
                    sample_rate: int = 16000,
                    dataset_path: str = None
                ):
        if instruments is None:
            instruments = self.INSTRUMENTS

        self.instruments = instruments
        self.duration = duration
        self.sample_rate = sample_rate

        # initialize the tinysol dataset
        metadata = os.path.join(dataset_path, 'TinySOL_metadata.csv')
        df = pd.read_csv(metadata) 
        paths = df.iloc[:, 0].apply(lambda x: os.path.join(dataset_path, x))
        paths = paths.values.tolist()
        labels = df.iloc[:, 4].values

        # make sure the instruments passed in are valid
        for instrument in instruments:
            assert instrument in self.instruments, f"{instrument} is not a valid instrument"

        # load all tracks for this instrument
        self.tracks = []
        for path, label in zip(paths, labels):
            if label in self.instruments:
              self.tracks.append([path, label])


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
    