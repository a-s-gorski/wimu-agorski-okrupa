import logging
import click

from src.types.config import DeploymentConfig
from src.utils.yaml_utils import load_yaml_data
from src.types.endpoint import SupportModel, QueryModel
import torch

from src.deployment.inference import load_model, run_inference, calculate_predictions


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path: str):

    logger = logging.getLogger(__name__)
    yaml_data = load_yaml_data(config_path)
    deployment_config = DeploymentConfig(**yaml_data)
    logger.info(f"Loaded deployment config with params: {deployment_config}")
    protonet = load_model(deployment_config).to("cpu")
    classlist_support = ['Organ', 'Flute', 'Trumpet']
    support = {
        'audio': torch.randn(size=[15, 1, 16000]).tolist(),
        'target': torch.tensor([0 for _ in range(5)] + [1 for _ in range(5)] + [2 for _ in range(5)]),
        'classlist': classlist_support,
    }
    query = {
        'audio': torch.randn(size=[45, 1, 16000]).tolist(),
    }

    s = SupportModel(**support)
    q = QueryModel(**query)

    logits = run_inference(protonet, s, q)
    predicted_labels, _ = calculate_predictions(logits, support=s)
    logging.info(f"Inference results: {predicted_labels}")


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()