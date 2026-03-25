from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import torch
from torch import nn
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat


CheckpointLike = Union[str, Path, Dict[str, Any]]


def extract_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return checkpoint.get("model_state_dict", checkpoint)


def checkpoint_uses_qat(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(
        ".weight_fake_quant." in name or ".activation_post_process." in name
        for name in state_dict
    )


def model_is_qat_prepared(model: nn.Module) -> bool:
    return any(hasattr(module, "weight_fake_quant") for module in model.modules())


def prepare_model_for_qat(model: nn.Module) -> nn.Module:
    if model_is_qat_prepared(model):
        return model

    model.train()
    model.qconfig = get_default_qat_qconfig("fbgemm")
    for module in model.modules():
        if isinstance(module, (nn.GRU, nn.Embedding)):
            module.qconfig = None
    prepare_qat(model, inplace=True)
    return model


def _load_checkpoint_object(checkpoint: CheckpointLike, map_location: str = "cpu") -> Dict[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        return torch.load(Path(checkpoint).as_posix(), map_location=map_location)
    return checkpoint


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint: CheckpointLike,
    *,
    strict: bool = True,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    checkpoint_obj = _load_checkpoint_object(checkpoint, map_location=map_location)
    state_dict = extract_state_dict(checkpoint_obj)
    was_training = model.training

    if checkpoint_uses_qat(state_dict):
        prepare_model_for_qat(model)

    model.load_state_dict(state_dict, strict=strict)
    if not was_training:
        model.eval()
    return checkpoint_obj

