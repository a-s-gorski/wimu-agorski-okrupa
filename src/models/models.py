from music_fsl.backbone import Backbone
from music_fsl.protonet import PrototypicalNet

from src.models.few_shot_learner import FewShotLearner
from src.types.config import TrainingConfig
from src.types.model import PrototypicalNetType, FewShotLearnerType


def get_protypical_net(config: TrainingConfig) -> PrototypicalNet:
    backbone = Backbone(sample_rate=config.sample_rate)
    prototypical_net = PrototypicalNetType[config.model_type].value
    protonet = prototypical_net(backbone)
    print(PrototypicalNetType[config.model_type].value)
    return protonet

def get_learner(config: TrainingConfig, protonet: PrototypicalNet) -> FewShotLearner:
    learner = FewShotLearnerType[config.model_type].value
    print(FewShotLearnerType[config.model_type].value)
    model = learner(protonet=protonet)
    return model
