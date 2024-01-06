import os
import librosa
from collections import defaultdict
from typing import Dict, List

import music_fsl.util as util

from src.data.class_conditional_dataset import ClassConditionalDataset

class GoodSounds(ClassConditionalDataset):
    """
    Initialize a `GoodSounds Dataset Loader` dataset instance.
    
    Args:
        instruments (List[str]): A list of instruments to include in the dataset.
        duration (float): The duration of each audio clip in the dataset (in seconds).
        sample_rate (int): The sample rate of the audio clips in the dataset (in Hz).
        dataset - loaded mirdata.dataset
    """

    INSTRUMENTS = [
        'flute', 'cello', 'clarinet', 'trumpet', 'violin', 'sax_alto', 'sax_tenor', 'sax_baritone', 'sax_soprano', 'oboe', 'piccolo', 'bass'
    ]

    TRAIN_INSTRUMENTS = [
        'cello', 'clarinet', 'violin', 'sax_alto', 'sax_baritone', 'sax_soprano', 'piccolo',
        ]

    TEST_INSTRUMENTS = [
        'flute', 'trumpet', 'sax_tenor', 'oboe', 'bass'
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
        self.dataset_path = dataset_path

        # make sure the instruments passed in are valid
        for instrument in instruments:
            assert instrument in self.INSTRUMENTS, f"{instrument} is not a valid instrument"

        # load all tracks for this instrument
        self.tracks = []
        for dir in os.listdir(self.dataset_path):
            ins = dir.split('_')[0]
            if ins in self.instruments:
                for subdir_dir, dirs_dir, files_dir in os.walk(os.path.join(self.dataset_path, dir, 'neumann')):
                    for file in files_dir:
                        if file.endswith('.wav'):
                            if librosa.get_duration(filename=os.path.join(self.dataset_path, dir, 'neumann', file)) >= duration:
                                self.tracks.append([os.path.join(self.dataset_path, dir, 'neumann', file), ins])
            else:
                ins = f'{ins}_{dir.split("_")[1]}'
                if ins in self.instruments:
                    for subdir_dir, dirs_dir, files_dir in os.walk(os.path.join(self.dataset_path, dir, 'neumann')):
                        for file in files_dir:
                            if file.endswith('.wav'):
                                if librosa.get_duration(filename=os.path.join(self.dataset_path, dir, 'neumann', file)) >= duration:
                                    self.tracks.append([os.path.join(self.dataset_path, dir, 'neumann', file), ins])

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
    