"""EfficientAd: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.

https://arxiv.org/pdf/2303.14535.pdf.
"""
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import torch
from pathlib import Path
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.models import EfficientAd

logger = logging.getLogger(__name__)


class VidEfficientAd(EfficientAd):
    """PL Lightning Module for the EfficientAd algorithm.

    Args:
        imagenet_dir (Path|str): directory path for the Imagenet dataset
            Defaults to ``./datasets/imagenette``.
        teacher_out_channels (int): number of convolution output channels
            Defaults to ``384``.
        model_size (str): size of student and teacher model
            Defaults to ``EfficientAdModelSize.S``.
        lr (float): learning rate
            Defaults to ``0.0001``.
        weight_decay (float): optimizer weight decay
            Defaults to ``0.00001``.
        padding (bool): use padding in convoluional layers
            Defaults to ``False``.
        pad_maps (bool): relevant if padding is set to False. In this case, pad_maps = True pads the
            output anomaly maps so that their size matches the size in the padding = True case.
            Defaults to ``True``.
    """

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """
        Perform the validation step of EfficientAd returns anomaly maps for the input image batch.
        Input shape: B x F x C x W x H
        Args:
          batch (dict[str, str | torch.Tensor]): Input batch
          args: Additional arguments.
          kwargs: Additional keyword arguments.

        Returns:
          Dictionary containing anomaly maps.
        """
        del args, kwargs  # These variables are not used.

        print(batch.keys())
        for b in range(batch["image"].shape[0]):
            batch["anomaly_maps"][b] = self.model(batch["image"][b])["anomaly_map"]

        return batch
