from src.models.models import get_protypical_net
from src.utils.yaml_utils import load_yaml_data
from src.types.config import TrainingConfig
from src.models.few_shot_learner import FewShotLearner
import torch
import torch
import random


config_path = "config/training.yml"
yaml_data = load_yaml_data(config_path)
training_config = TrainingConfig(**yaml_data)
protype_net = get_protypical_net(config=training_config)
model = FewShotLearner.load_from_checkpoint("deployments/models/epoch=9-step=10000.ckpt")
model = model.to('cpu')
model.eval()
# audio = torch.randn(size=(15, 1, 16000))
# offset = []

# # support, query = batch

# app = FastAPI()

# @app.get("/predict/")

classlist_support = ['Organ', 'Flute', 'Trumpet']
classlist_query = ['Cello', 'Human singing voice', 'Clarinet']

support = {
    'audio': torch.randn(size=[15, 1, 16000]),
    'offset': [random.random() for _ in range(15)],
    'duration': [1.0 for _ in range(15)],
    'label': [random.choice(classlist_support) for _ in range(15)],
    'target': torch.randn(size=[15, ],).to(torch.long),
    'classlist': classlist_support,
}
query = {
    'audio': torch.randn(size=[60, 1, 16000]),
    'offset': [random.random() for _ in range(60)],
    'duration': [1.0 for _ in range(60)],
    'label': [random.choice(classlist_query) for _ in range(60)],
    'target': torch.randn(size=[60, ]).to(torch.long),
    'classlist': classlist_query,
}

print(model.step([support, query], 0, "predictions"))

# KEY audio
# torch.Size([15, 1, 16000])
# KEY offset
# 15 [0.13850829215401814, 0.3706013903443883, 1.8891781934091343, 0.5327027868962979, 1.4920655508365108, 0.7918853295390722, 0.27770542371381446, 1.7413801627081256, 1.4292280822146142, 0.9035840347367027, 0.4150741218814071, 0.5008036942717418, 1.1176159853132646, 0.749590120416868, 1.4420810087874678]
# KEY duration
# 15 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# KEY label
# 15 ['Organ', 'Organ', 'Organ', 'Organ', 'Organ', 'Flute', 'Flute', 'Flute', 'Flute', 'Flute', 'Trumpet', 'Trumpet', 'Trumpet', 'Trumpet', 'Trumpet']
# KEY target
# torch.Size([15])
# KEY classlist
# 3 ['Organ', 'Flute', 'Trumpet']
# -----
# audio <class 'torch.Tensor'>
# offset <class 'list'>
# duration <class 'list'>
# label <class 'list'>
# target <class 'torch.Tensor'>
# classlist <class 'list'>