from dataclasses import dataclass, field, fields
from typing import Union
import os
import yaml


@dataclass
class ModelConfig:
    name: str = ""
    device: str = "cuda"
    dtype: str = "bfloat16"
    dummy: bool = False


@dataclass
class DataConfig:
    dataset_name: str = "open_thoughts"
    num_samples: int = 4096
    val_samples: int = 128
    batch_size: int = 64
    max_length: int = 4096
    num_workers: int = 8
    seed: int = 0
    shuffle_seed: int = 1234
    shuffle_buffer_size: int = 100_000
    open_thoughts_max_samples: int = 10_000


@dataclass
class QuantizationConfig:
    gsq_bits: Union[int, float, str] = 2
    init_method: str = "gptq"
    gsq_enabled: bool = True
    start_layer: int = 0
    self_attn: bool = False
    std: float = 0.01
    temperature: list = field(default_factory=lambda: [2, 0.05])
    scale: list = field(default_factory=lambda: [100, 500])
    groupsize: int = 128
    strength: float = 6
    logits_dtype: str = "bfloat16"


@dataclass
class TrainingConfig:
    num_epochs: int = 10
    device_microbatch_size: int = 16
    lr1: float = 0.0001
    lr2: float = 0.00005
    weight_decay: float = 1.0
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    eval_baseline: bool = True
    lion_betas: list = field(default_factory=lambda: [0.9, 0.95])
    warmup_steps: int = 0
    lr_decay_type: str = "cosine"
    scheduler_min_lr: float = 0.1
    meta_init_std: float = 0.02
    act_cache_mmap_threshold_gb: float = 2.0
    act_cache_dir: str = "gsq/act_cache"
    ppl_eval_every_n_layers: int = 6


@dataclass
class GPTQConfig:
    nsamples: int = 512
    wbits: int = 2
    sym: bool = True
    trits: bool = False
    percdamp: float = 0.01
    blocksize: int = 128
    groupsize: int = 128
    static_groups: bool = False
    prunen: int = 0
    prunem: int = 0


@dataclass
class EvalConfig:
    default_tasks: str = "gsm8k,arc_challenge,arc_easy,winogrande,piqa"
    ppl_seed: int = 1234
    ppl_max_samples: int = 1000
    test_size: float = 0.2
    split_seed: int = 42


@dataclass
class DistributedConfig:
    timeout_hours: float = 2.0


@dataclass
class WandBConfig:
    enabled: bool = True
    project: str = ""
    entity: str = ""


@dataclass
class LoggingConfig:
    gumbel_step_log_interval: int = 50
    step_report_divisor: int = 5
    calib_report_divisor: int = 10
    ppl_report_divisor: int = 10
    batch_report_divisor: int = 5


@dataclass
class GSQConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gptq: GPTQConfig = field(default_factory=GPTQConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _build_dataclass(cls, raw):
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict for {cls.__name__}, got {type(raw).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) in '{cls.__name__}' section: {sorted(unknown)}. "
            f"Allowed keys: {sorted(known)}"
        )
    return cls(**raw)


_YAML_KEY_TO_FIELD = {
    "model": "model",
    "data": "data",
    "quantization": "quantization",
    "training": "training",
    "gptq": "gptq",
    "eval": "eval",
    "distributed": "distributed",
    "wandb": "wandb",
    "logging": "logging",
}

_FIELD_TO_DATACLASS = {
    "model": ModelConfig,
    "data": DataConfig,
    "quantization": QuantizationConfig,
    "training": TrainingConfig,
    "gptq": GPTQConfig,
    "eval": EvalConfig,
    "distributed": DistributedConfig,
    "wandb": WandBConfig,
    "logging": LoggingConfig,
}


def _expand_placeholders(tree):
    if isinstance(tree, str):
        return os.path.expandvars(tree)
    if isinstance(tree, dict):
        return {k: _expand_placeholders(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_expand_placeholders(v) for v in tree]
    return tree


def _apply_env_overrides(cfg):
    if "GSQ_MODEL_NAME" in os.environ:
        cfg.model.name = os.path.expandvars(os.environ["GSQ_MODEL_NAME"])
    if "GSQ_CHECKPOINT_DIR" in os.environ:
        cfg.training.checkpoint_dir = os.path.expandvars(os.environ["GSQ_CHECKPOINT_DIR"])
    if "GSQ_LOG_DIR" in os.environ:
        cfg.training.log_dir = os.path.expandvars(os.environ["GSQ_LOG_DIR"])
    if "GSQ_ACT_CACHE_DIR" in os.environ:
        cfg.training.act_cache_dir = os.path.expandvars(os.environ["GSQ_ACT_CACHE_DIR"])
    if "WANDB_PROJECT" in os.environ:
        cfg.wandb.project = os.path.expandvars(os.environ["WANDB_PROJECT"])
    elif not cfg.wandb.project.strip():
        cfg.wandb.project = "gsq"
    if "WANDB_ENTITY" in os.environ:
        cfg.wandb.entity = os.path.expandvars(os.environ["WANDB_ENTITY"])


def load_config(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    raw = _expand_placeholders(raw)

    if not isinstance(raw, dict):
        raise TypeError(f"Expected top-level YAML mapping, got {type(raw).__name__}")

    # Backward compat: bare `wandb: true/false` -> WandBConfig
    if "wandb" in raw and not isinstance(raw["wandb"], dict):
        raw["wandb"] = {"enabled": bool(raw["wandb"])}

    unknown_top = set(raw) - set(_YAML_KEY_TO_FIELD)
    if unknown_top:
        raise ValueError(
            f"Unknown top-level config section(s): {sorted(unknown_top)}. "
            f"Allowed sections: {sorted(_YAML_KEY_TO_FIELD)}"
        )

    kwargs = {}
    for yaml_key, field_name in _YAML_KEY_TO_FIELD.items():
        section = raw.get(yaml_key, None)
        cls = _FIELD_TO_DATACLASS[field_name]
        kwargs[field_name] = _build_dataclass(cls, section)

    cfg = GSQConfig(**kwargs)

    _apply_env_overrides(cfg)

    return cfg
