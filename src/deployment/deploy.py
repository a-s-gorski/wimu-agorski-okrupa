from fastapi import FastAPI
import os
import logging
import click
import uvicorn

from src.types.endpoint import SupportModel, QueryModel, PredictOutput
from src.deployment.inference import load_model, calculate_predictions, run_inference
from src.types.config import DeploymentConfig
from src.utils.yaml_utils import load_yaml_data

app = FastAPI()
config_path = os.getenv("CONFIG_PATH")
yaml_data = load_yaml_data(config_path)
deployment_config = DeploymentConfig(**yaml_data)
model = load_model(deployment_config)
model = model.to("cpu")

@app.get("/predict", response_model=PredictOutput)
def predict(support: SupportModel, query: QueryModel):
    logits = run_inference(model, support, query)
    predicted_labels, predicted_classes = calculate_predictions(logits, support=support)

    return PredictOutput(logits=logits, predicted_labels=predicted_labels, predicted_classes=predicted_classes)


@click.command()
@click.argument('url', type=str)
@click.argument('port', type=int)
def main(url: str, port: int):
    uvicorn.run(app, host=url, port=port, reload=False)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
