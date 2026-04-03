"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from model_utils.processors.base_processor import BaseProcessor
from model_utils.processors.blip_processors import BlipCaptionProcessor
from model_utils.processors.clip_processors import (
    ClipImageTrainProcessor,
    ClipImageEvalProcessor,
)

from model_utils.common.registry import registry

__all__ = [
    "BaseProcessor",
    "BlipCaptionProcessor",
    "ClipImageTrainProcessor",
    "ClipImageEvalProcessor",
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
