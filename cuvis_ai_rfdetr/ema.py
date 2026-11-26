"""RF-DETR's weight averaging (EMA), one update at a time.

RF-DETR's trainer keeps a ``torch.optim.swa_utils.AveragedModel`` with
``use_buffers=True`` and this average function: ``updates = num_averaged + 1``,
``d = decay * (1 - exp(-updates / tau))`` (``d = decay`` when ``tau == 0``),
``avg = avg * d + live * (1 - d)``. The first update copies the live weights.
It updates once per optimizer step and once more at the end of every training epoch.
"""

from __future__ import annotations

import math

import torch
from torch import nn


@torch.no_grad()
def ema_update_(
    averaged: nn.Module, model: nn.Module, num_averaged: int, decay: float, tau: float
) -> None:
    """Apply one RF-DETR EMA update of ``averaged`` towards ``model`` in place.

    ``num_averaged`` is the number of updates applied before this one (0 on the first
    call, which copies). Parameters and floating-point buffers are averaged; other
    buffers are copied.
    """
    pairs = list(zip(averaged.parameters(), model.parameters(), strict=True))
    pairs += list(zip(averaged.buffers(), model.buffers(), strict=True))
    if num_averaged == 0:
        for avg, live in pairs:
            avg.copy_(live.detach())
        return
    updates = int(num_averaged) + 1
    if tau > 0:
        d = float(decay) * (1.0 - math.exp(-updates / float(tau)))
    else:
        d = float(decay)
    for avg, live in pairs:
        if avg.is_floating_point():
            avg.copy_(avg * d + live.detach() * (1.0 - d))
        else:
            avg.copy_(live.detach())
