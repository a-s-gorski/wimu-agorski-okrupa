from typing import Tuple, List
from src.types.config import DeploymentConfig
from src.types.model import FewShotLearnerType
from src.types.endpoint import SupportModel, QueryModel
from torch import nn
import torch
from music_fsl.protonet import PrototypicalNet
from src.models.prototypical_net_ksoft import PrototypicalNetKSoft
from src.models.prototypical_net_ksoft_distractor import PrototypicalNetKSoftWithDistractor


def load_model(deployment_config: DeploymentConfig) -> nn.Module:
    learner = FewShotLearnerType[deployment_config._model_type].value
    model = learner.load_from_checkpoint(deployment_config.checkpoint_model_path)
    protonet = model.protonet
    return protonet


def run_inference(model: PrototypicalNet | PrototypicalNetKSoft | PrototypicalNetKSoftWithDistractor, support: SupportModel, query: QueryModel) -> torch.Tensor:
    support_dict = support.model_dump()
    query_dict = query.model_dump()
    support_dict["audio"] = torch.tensor(support_dict["audio"])
    support_dict["target"] = torch.tensor(support_dict["target"])
    query_dict["audio"] = torch.tensor(query_dict["audio"])

    logits = model(support_dict, query_dict)
    return logits


def calculate_predictions(logits: torch.Tensor, support: SupportModel) -> Tuple[List[int], List[str]]:
    predicted_labels = torch.argmax(logits, dim=1).tolist()
    predicted_classes = list(map(lambda id: support.classlist[id], predicted_labels))
    return predicted_labels, predicted_classes
