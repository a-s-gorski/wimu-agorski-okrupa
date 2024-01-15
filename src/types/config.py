from typing import Literal, Optional

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    sample_rate: Optional[int] = 16000
    n_way: Optional[int] = 3
    n_support: Optional[int] = 5
    n_query: Optional[int] = 20
    n_unlabeled: Optional[int] = 0
    n_distractor: Optional[int] = 0
    n_train_episodes: Optional[int] = 1000
    n_val_episodes: Optional[int] = 50
    num_workers: Optional[int] = -1
    model_type: Optional[Literal["protonet", "softkmeans", "softkmeansdistractor"]] = "protonet"

    def __repr__(self):
        repr_dict = {k: v for k, v in self.model_dump().items()
                     if v is not None}
        return f"{self.__class__.__name__}({repr_dict})"


class TrainingConfig(BaseModel):
    sample_rate: Optional[int] = 16000
    max_epochs: Optional[int] = 1
    log_every_n_steps: Optional[int] = 1
    val_check_interval: Optional[int] = 50
    profiler: Optional[Literal["simple"]] = "simple"
    logger: Optional[Literal["tensorboard"]] = "tensorboard"
    model_type: Optional[Literal["protonet", "softkmeans", "softkmeansdistractor"]] = "protonet"

    def __repr__(self):
        repr_dict = {k: v for k, v in self.model_dump().items()
                     if v is not None}
        return f"{self.__class__.__name__}({repr_dict})"
