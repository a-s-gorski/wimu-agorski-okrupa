from pydantic import BaseModel, model_validator, validator
from typing import List
import torch
from collections import Counter


class SupportModel(BaseModel):
    audio: List[List[List[float]]]
    target: List[int]
    classlist: List[str]

    @validator("audio")
    def validate_audio(cls, audio: List[List[List[float]]]):
        audio_tensor = torch.tensor(audio)
        if audio_tensor.shape[1] != 1:
            raise ValueError(f"Expected mono support audio, got {audio_tensor.shape[1]}")
        return audio

    @validator("target")
    def validate_target(cls, target: List[float]):
        counted_dict = dict(Counter(target))
        if not len(set(target_class == target[0] for target_class in counted_dict.values())) == 1:
            raise ValueError("All target classes in support query have to have the same distributions.")
        return target

    @model_validator(mode='before')
    @classmethod
    def validate_batchsize(cls, values):
        audio, target = values.get('audio'), values.get('target')
        audio_tensor = torch.tensor(audio)
        if audio_tensor.shape[0] != len(target):
            raise ValueError(f"Batch size of audio_tensor {audio_tensor.shape[0]} is not equal \
                             to batch_size of target {len(target)}")
        return values


class QueryModel(BaseModel):
    audio: List[List[List[float]]]

    @validator("audio")
    def validate_audio(cls, audio: List[float]):
        audio_tensor = torch.tensor(audio)
        if len(audio_tensor.shape) != 3:
            raise ValueError(f"Support audio expects 3 dimensions got {len(audio_tensor.shape)}")
        if audio_tensor.shape[1] != 1:
            raise ValueError(f"Expected mono support audio, got {audio_tensor.shape[1]}")
        return audio


class PredictOutput(BaseModel):
    logits: List[List[float]]
    predicted_labels: List[int]
    predicted_classes: List[str]
