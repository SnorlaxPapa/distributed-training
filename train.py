import contextlib
import os
import signal
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import timedelta
from typing import Any, cast

import torch
from loguru import logger
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record

from internals.components.checkpoint import CheckpointManager
from internals.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from internals.components.loss import (
    IGNORE_INDEX,
    LossFunction,
    build_cross_entropy_loss,
)
from internals.components.lr_scheduler import (
    LRSchedulersContainer,
    build_lr_schedulers,
)
from internals.components.metrics import (
    MetricsProcessor,
    collect_parameter_norm_metrics,
)
from internals.components.optimizer import (
    OptimizersContainer,
    build_optimizers_with_moe_load_balancing,
)
from internals.components.tokenizer import (
    DeepSeekV3Tokenizer,
)
from internals.config import TORCH_DTYPE_MAP, JobConfig
from internals.config.default_configs import (
    get_config,
)
from internals.config.job_config import Parallelism
from internals.datasets.hf_datasets import build_hf_dataloader
from internals.distributed import ParallelDims
from internals.distributed import utils as dist_utils
from internals.distributed.pipeline_parallel import pipeline_llm
from internals.model.model import DeepSeekV3Model
from internals.model.parallelize import parallelize_deepseekv3
from internals.tools import device_utils, utils
from internals.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)


class Trainer(Stateful):
    job_config: JobConfig
    parallel_dims: ParallelDims

    tokenizer: DeepSeekV3Tokenizer
    dataloader: BaseDataLoader
    model_parts: list[torch.nn.Module]
    loss_fn: LossFunction
    optimizers: OptimizersContainer
    lr_schedulers : LRSchedulersContainer
    metrics_processor: MetricsProcessor
    checkpointer: CheckpointManager

    device: torch.device
    gc_handler: utils.GarbageCollection
    train_content: Callable[..., contextlib.AbstractContextManager]
    maybe_enable_map: contextlib.AbstractContextManager
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    step: int
    ntokens_seen: int