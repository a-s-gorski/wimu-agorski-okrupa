from music_fsl.backbone import Backbone
from music_fsl.protonet import PrototypicalNet

from src.types.config import TrainingConfig


def get_protypical_net(config: TrainingConfig) -> PrototypicalNet:
    backbone = Backbone(sample_rate=config.sample_rate)
    protonet = PrototypicalNet(backbone)
    return protonet
