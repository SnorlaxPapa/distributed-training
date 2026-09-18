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

from distributed_training.components.checkpoint import CheckpointManager
from distributed_training.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from distributed_training.components.loss import (
    IGNORE_INDEX,
    LossFunction,
    build_cross_entropy_loss,
)
from distributed_training.components.lr_scheduler import (
    LRSchedulersContainer,
    build_lr_schedulers,
)
from distributed_training.components.metrics import (
    MetricsProcessor,
    collect_parameter_norm_metrics,
)
from distributed_training.components.optimizer import (
    OptimizersContainer,
    build_optimizers_with_moe_load_balancing,
)
from distributed_training.components.tokenizer import (
    DeepSeekV3Tokenizer,
)
from distributed_training.config import TORCH_DTYPE_MAP, JobConfig
from distributed_training.config.default_configs import (
    get_config,
)
from distributed_training.config.job_config import Parallelism
from distributed_training.datasets.hf_datasets import build_hf_dataloader
from distributed_training.distributed import ParallelDims
from distributed_training.distributed import utils as dist_utils
from distributed_training.distributed.pipeline_parallel import pipeline_llm
from distributed_training.model.model import DeepSeekV3Model
from distributed_training.model.parallelize import parallelize_deepseekv3
from distributed_training.tools import device_utils, utils
from tdistributed_training.tools.profiling import (
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