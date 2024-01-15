from enum import Enum
from music_fsl.protonet import PrototypicalNet

from src.data.episode_dataset import EpisodeDataset, EpisodeDatasetUnlabeled, EpisodeDatasetUnlabeledWithDistractor
from src.models.few_shot_learner import FewShotLearner
from src.models.few_shot_learner_ksoft import FewShotLearnerKSoft
from src.models.few_shot_learner_ksoft_distractor import FewShotLearnerUnlabeledWithDistractor
from src.models.prototypical_net_ksoft import PrototypicalNetKSoft
from src.models.prototypical_net_ksoft_distractor import PrototypicalNetKSoftWithDistractor


class ModelEpisodeType(Enum):
    protonet = EpisodeDataset
    softkmeans = EpisodeDatasetUnlabeled
    softkmeansdistractor = EpisodeDatasetUnlabeledWithDistractor

class PrototypicalNetType(Enum):
    protonet = PrototypicalNet
    softkmeans = PrototypicalNetKSoft
    softkmeansdistractor = PrototypicalNetKSoftWithDistractor

class FewShotLearnerType(Enum):
    protonet = FewShotLearner
    softkmeans = FewShotLearnerKSoft
    softkmeansdistractor = FewShotLearnerUnlabeledWithDistractor
