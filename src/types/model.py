from enum import Enum
from music_fsl.protonet import PrototypicalNet

from src.data.episode_dataset import EpisodeDataset, EpisodeDatasetUnlabeled
from src.models.few_shot_learner import FewShotLearner
from src.models.few_shot_learner_ksoft import FewShotLearnerKSoft
from src.models.prototypical_net_ksoft import PrototypicalNetKSoft


class ModelEpisodeType(Enum):
    protonet = EpisodeDataset
    softkmeans = EpisodeDatasetUnlabeled

class PrototypicalNetType(Enum):
    protonet = PrototypicalNet
    softkmeans = PrototypicalNetKSoft

class FewShotLearnerType(Enum):
    protonet = FewShotLearner
    softkmeans = FewShotLearnerKSoft
