import os
import pandas as pd
from collections import defaultdict
from typing import Dict, List

import music_fsl.util as util

from src.data.class_conditional_dataset import ClassConditionalDataset

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

    TRAIN_INSTRUMENTS = [
            'flute', 'piano', 'tenor saxophone', 'trumpet', 'violin'
        ]

    TEST_INSTRUMENTS = [
            'clarinet', 'distorted electric guitar', 'female singer',
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
