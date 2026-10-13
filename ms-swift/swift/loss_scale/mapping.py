# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import ALL_BASE_STRATEGY, LossScale

loss_scale_map = {'base': LossScale}


def get_loss_scale(loss_scale: str) -> LossScale:
    splited = loss_scale.split('+', 1)
    if len(splited) == 1:
        if splited[0] in ALL_BASE_STRATEGY:
            base_strategy, loss_scale = splited[0], 'base'
        else:
            base_strategy, loss_scale = 'default', splited[0]
    else:
        base_strategy, loss_scale = splited
    return loss_scale_map[loss_scale](base_strategy)
