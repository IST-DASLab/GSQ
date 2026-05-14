import os, sys, gc, json
import argparse
import yaml
from datetime import timedelta
import torch
import torch.nn as nn
import wandb
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(**_kwargs): pass
from transformers import AutoTokenizer
from src.config import load_config
from src.data.dataset import create_dataloader
from src.trainer import QuantizationTrainer
from src.utils.logging_utils import QuantizationLogger
from src.utils import progress_reporter
from src.utils.progress_reporter import (
    report_layer, report_gptq_done, report_throughput, report_pipeline,
)
import threading
import time
import uuid
import torch.distributed as dist


GLOBAL_RANK = 0

PROGRESS_FILENAME = "progress.json"


def _distributed_destroy_pg():
    if not dist.is_initialized():
        return
    skip = os.environ.get("GSQ_SKIP_DIST_DESTROY", "").strip().lower() in ("1", "true", "yes")
    if skip:
        if GLOBAL_RANK == 0:
            report_pipeline(
                "Skipping destroy_process_group() (GSQ_SKIP_DIST_DESTROY); "
                "launcher may briefly wait on child processes."
            )
        return
    raw = os.environ.get("GSQ_DIST_DESTROY_TIMEOUT_SEC", "120")

    try:
        timeout_sec = float(raw)
    except ValueError:
        timeout_sec = 120.0

    def attempt_destroy():
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    if timeout_sec <= 0:
        attempt_destroy()
        return

    thread = threading.Thread(target=attempt_destroy, name="destroy_process_group")
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        if GLOBAL_RANK == 0:
            report_pipeline(
                f"destroy_process_group blocked ≥{timeout_sec:.0f}s — forcing exit "
                "(NCCL/Python known issue); use GSQ_DIST_DESTROY_TIMEOUT_SEC=0 to wait "
                "forever, or GSQ_SKIP_DIST_DESTROY=1 to omit destroy."
            )
        os._exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Model Quantization Training')
    parser.add_argument('--config', type=str, default='configs/local/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--resume', type=str, nargs='?', const='latest', default=None,
                      help='Resume training. Pass a run_id to resume a specific run, '
                           'or omit the value to resume the latest run.')
    parser.add_argument('--max-layers', type=int, default=None,
                      help='Stop after quantizing this many layers (for smoke tests).')
    return parser.parse_args()


def generate_run_id():
    ts = time.strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{short}"


def _run_dir(checkpoint_dir, run_id):
    return os.path.join(checkpoint_dir, run_id)


def _progress_path(run_dir):
    return os.path.join(run_dir, PROGRESS_FILENAME)


def find_latest_run(checkpoint_dir):
    """Find the most recent run directory that has a progress.json."""
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for name in os.listdir(checkpoint_dir):
        run_path = os.path.join(checkpoint_dir, name)
        prog_path = os.path.join(run_path, PROGRESS_FILENAME)
        if os.path.isdir(run_path) and os.path.exists(prog_path):
            mtime = os.path.getmtime(prog_path)
            candidates.append((mtime, name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_progress(run_dir):
    path = _progress_path(run_dir)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def save_progress(run_dir, layer_idx, run_id=None, wandb_run_id=None):
    os.makedirs(run_dir, exist_ok=True)
    progress = {
        "run_id": run_id,
        "last_completed_layer": layer_idx,
        "wandb_run_id": wandb_run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = _progress_path(run_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, path)

def create_dict(num_samples, seqlen, hidden_size, dtype, mmap_dir=None, mmap_threshold_bytes=2 * 1024 ** 3):
    nbytes = num_samples * seqlen * hidden_size * 2  # 2 bytes for any 16-bit dtype
    if mmap_dir is not None and nbytes > mmap_threshold_bytes:
        import numpy as np
        os.makedirs(mmap_dir, exist_ok=True)
        path = os.path.join(mmap_dir, f"act_{os.getpid()}_{num_samples}x{seqlen}x{hidden_size}.bin")
        arr = np.memmap(path, dtype='uint16', mode='w+', shape=(num_samples, seqlen, hidden_size))
        inps = torch.from_numpy(arr).view(dtype)
        return {'input': inps, '_mmap_path': path}
    inps = torch.zeros((num_samples, seqlen, hidden_size), dtype=dtype, device='cpu')
    return {'input': inps}


def _cleanup_act_cache_mmap(*dicts):
    for d in dicts:
        path = d.pop('_mmap_path', None)
        if path is not None and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass

def resolve_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def train_all_layers(model, train_loader, val_loader, gpt_loader, logger, config,
                     resume_from_layer=-1, run_id=None, max_layers=None):
    """Train all layers, optionally resuming after a previously completed layer.

    Args:
        resume_from_layer: index of the last *completed* layer (-1 = fresh start).
            Layers 0..resume_from_layer will be replayed (activations propagated
            through saved weights) without re-training.
        run_id: unique identifier for this run, stored in progress.json.
        max_layers: if set, stop after quantizing this many layers (smoke test).
    """
    global GLOBAL_RANK
    device = model.device

    dtype = model.dtype
    model.save_dir = config.training.checkpoint_dir
    current_layer = model.get_current_layer()
    # Resolve text-model config (multimodal models nest it under text_config)
    _cfg = model.model.config
    if hasattr(_cfg, "text_config"):
        _cfg = _cfg.text_config
    first_k_dense_replace = getattr(_cfg, "first_k_dense_replace", 0)
    hidden_size = _cfg.hidden_size

    first_layer_was_trained = (resume_from_layer >= 0
                               and 1 > first_k_dense_replace
                               and 1 > config.quantization.start_layer)
    if first_layer_was_trained:
        model.load_from_disc(current_layer)
    elif config.quantization.start_layer == 0 or first_k_dense_replace > 0:
        if GLOBAL_RANK == 0:
            logger.logger.info(f"Loading weights for layer: {current_layer}")
        model.move_layer_to_gpu(current_layer)
    else:
        model.load_from_disc(current_layer)
    if GLOBAL_RANK == 0:
        logger.logger.info("Loading embedding weights")
    model.move_embed_to(device)

    mmap_dir = config.training.act_cache_dir or None
    mmap_threshold = int(config.training.act_cache_mmap_threshold_gb * 1024 ** 3)
    train_all = create_dict(len(train_loader.dataset), model.model.seqlen, hidden_size, dtype, mmap_dir, mmap_threshold)
    val_all = create_dict(len(val_loader.dataset), model.model.seqlen, hidden_size, dtype, mmap_dir, mmap_threshold)
    gpt_all = create_dict(len(gpt_loader.dataset), model.model.seqlen, hidden_size, dtype, mmap_dir, mmap_threshold)

    try:
        if GLOBAL_RANK == 0:
            logger.logger.info(f"Capturing GPTQ inputs ({len(gpt_loader.dataset)} samples)")
        model.get_inputs(gpt_all, gpt_loader)
        if GLOBAL_RANK == 0:
            logger.logger.info(f"Capturing train inputs ({len(train_loader.dataset)} samples)")
        model.get_inputs(train_all, train_loader)
        if GLOBAL_RANK == 0:
            logger.logger.info(f"Capturing val inputs ({len(val_loader.dataset)} samples)")
        model.get_inputs(val_all, val_loader)

        if GLOBAL_RANK == 0:
            logger.logger.info("Offloading embedding to meta")
        model.move_embed_to('meta')

        num_layers = model.num_layers if hasattr(model, 'num_layers') else 0
        wandb_run_id = wandb.run.id if (config.wandb.enabled and GLOBAL_RANK == 0 and wandb.run) else None
        num_experts = getattr(model, 'num_experts', None)
        if resume_from_layer < 0 and config.training.eval_baseline:
            if GLOBAL_RANK == 0:
                logger.logger.info("Measuring baseline (dense) perplexity before quantization")
            baseline_ppl = model.ppl_evaluation(-1)
            if GLOBAL_RANK == 0:
                logger.logger.info(f"eval/baseline_ppl: {baseline_ppl:.4f}")
            if config.wandb.enabled and GLOBAL_RANK == 0:
                wandb.log({"eval/baseline_ppl": baseline_ppl}, step=0)
            if config.quantization.start_layer == 0 or first_k_dense_replace > 0:
                model.move_layer_to_gpu(current_layer)
            else:
                model.load_from_disc(current_layer)
            model.move_embed_to('meta')

        run_start = time.time()
        layer_times = []
        gptq_times = []
        train_times = []
        gsq_enabled = config.quantization.gsq_enabled
        count = 0
        while current_layer is not None:
            count += 1
            layer_idx = count - 1
            is_already_done = layer_idx <= resume_from_layer

            needs_training = (count > first_k_dense_replace
                              and count > config.quantization.start_layer)

            if needs_training and not is_already_done:
                if GLOBAL_RANK == 0:
                    init_label = config.quantization.init_method.upper()
                    logger.logger.info(f"Starting {'quantization' if gsq_enabled else init_label + '-only'} for layer: {current_layer}")

                if dist.is_initialized():
                    dist.barrier()
                layer_start = time.time()
                elapsed_total = layer_start - run_start

                if GLOBAL_RANK == 0:
                    report_layer(layer_idx, num_layers, config.quantization.init_method.upper(), elapsed_total)

                if config.quantization.self_attn and not model.is_moe:
                    trainer = QuantizationTrainer(model, config, dtype, self_attn=True)
                    model.get_layer_initialization(trainer, gpt_all, config, logger, is_attn=True)
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    if gsq_enabled:
                        trainer.train_layer(current_layer, train_all, val_all, logger,
                                            layer_idx=layer_idx, num_layers=num_layers)
                    del trainer
                    if GLOBAL_RANK == 0:
                        model.save_prefixes_to_disc(model._layer_prefixes(current_layer)['non_mlp'], exclude=["self_attn"])
                        model.save_attention_to_disc(f"{model.layer_prefix}.{count-1}.self_attn")
                else:
                    if GLOBAL_RANK == 0:
                        model.save_prefixes_to_disc(model._layer_prefixes(current_layer)['non_mlp'])

                trainer = QuantizationTrainer(model, config, dtype)
                gptq_start = time.time()
                model.get_layer_initialization(trainer, gpt_all, config, logger)
                torch.cuda.synchronize()
                gptq_time = time.time() - gptq_start
                gptq_times.append(gptq_time)

                if GLOBAL_RANK == 0:
                    gptq_avg_loss_tmp = getattr(trainer, 'gptq_avg_loss', None)
                    report_gptq_done(gptq_time, avg_loss=gptq_avg_loss_tmp)

                if gsq_enabled:
                    if model.is_moe:
                        model.get_mlp_input_all(train_all)
                        model.get_mlp_input_all(val_all)

                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()

                    if GLOBAL_RANK == 0:
                        report_layer(layer_idx, num_layers, "Gumbel-Softmax",
                                     time.time() - run_start)

                    train_start = time.time()
                    trainer.train_layer(current_layer, train_all, val_all, logger,
                                        layer_idx=layer_idx, num_layers=num_layers)
                    torch.cuda.synchronize()
                    train_time = time.time() - train_start
                    train_times.append(train_time)
                else:
                    train_time = 0.0
                    if model.is_moe:
                        model.save_moe_experts_to_disc()
                    elif GLOBAL_RANK == 0:
                        mlp_pfx = f"{model.layer_prefix}.{count-1}.mlp"
                        model.save_mlp_to_disc(mlp_pfx)

                model.temp_weights.clear()

                gptq_avg_loss = getattr(trainer, 'gptq_avg_loss', None)
                del trainer
                torch.cuda.empty_cache()

                if dist.is_initialized():
                    if GLOBAL_RANK == 0:
                        report_pipeline("Syncing ranks after trainer teardown (checkpoint phase)")
                    dist.barrier()
                layer_time = time.time() - layer_start
                layer_times.append(layer_time)
                avg_layer_time = sum(layer_times) / len(layer_times)

                if GLOBAL_RANK == 0:
                    report_throughput(
                        layer_idx, num_layers, layer_time, gptq_time, train_time,
                        num_experts=num_experts if model.is_moe else None,
                        avg_layer_time=avg_layer_time,
                    )

                if config.wandb.enabled and GLOBAL_RANK == 0:
                    mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                    layer_logs = {
                        "layer/index": layer_idx,
                        "layer/progress": layer_idx / max(num_layers, 1),
                        "timing/gptq_init_sec": gptq_time,
                        "timing/training_sec": train_time,
                        "timing/layer_total_sec": layer_time,
                        "timing/gptq_phase_sec_avg": sum(gptq_times) / len(gptq_times),
                        "timing/layer_sec_avg": avg_layer_time,
                        "throughput/layers_per_hour": 3600.0 / avg_layer_time if avg_layer_time > 0 else 0,
                        "gpu/max_memory_allocated_gb": mem_alloc,
                        "gpu/max_memory_reserved_gb": mem_reserved,
                    }
                    if train_times:
                        layer_logs["timing/gumbel_phase_sec_avg"] = sum(train_times) / len(train_times)
                    if model.is_moe and num_experts:
                        layer_logs["timing/sec_per_expert"] = layer_time / num_experts
                        layer_logs["throughput/experts_per_hour"] = (num_experts * 3600.0 / layer_time) if layer_time > 0 else 0
                    if gptq_avg_loss is not None:
                        layer_logs["gptq/avg_loss"] = gptq_avg_loss
                    wandb.log(layer_logs)
                    torch.cuda.reset_peak_memory_stats()

                _ppl_every = config.training.ppl_eval_every_n_layers
                if _ppl_every > 0 and (count - 1) % _ppl_every == 0:
                    if GLOBAL_RANK == 0:
                        report_pipeline(
                            "Running perplexity evaluation (full model forward; sparse console output)"
                        )
                    _ppl_t0 = time.time()
                    dataset_ppl = model.ppl_evaluation(count - 1)
                    if GLOBAL_RANK == 0:
                        logger.logger.info(f"eval/ppl (layer {layer_idx}): {dataset_ppl:.4f}")
                        report_pipeline(f"PPL evaluation done in {time.time() - _ppl_t0:.1f}s")
                    if config.wandb.enabled and GLOBAL_RANK == 0:
                        wandb.log({"eval/ppl": dataset_ppl, "layer/index": layer_idx})

                if GLOBAL_RANK == 0:
                    report_pipeline(f"Reloading quantized weights from disk for {current_layer}")
                _load_t0 = time.time()
                model.load_from_disc(current_layer)
                if GLOBAL_RANK == 0:
                    logger.logger.info(
                        f"Loaded quantized weights for {current_layer} in {time.time() - _load_t0:.1f}s"
                    )

                if GLOBAL_RANK == 0:
                    report_pipeline("Propagating train/val activations through quantized layer")
                if model.is_moe:
                    if gsq_enabled:
                        model.get_mlp_output_all(train_all)
                        model.get_mlp_output_all(val_all)
                    else:
                        model.get_layer_activations(train_all)
                        model.get_layer_activations(val_all)
                else:
                    model.get_layer_activations(train_all)
                    model.get_layer_activations(val_all)

                if GLOBAL_RANK == 0:
                    report_pipeline("Writing progress.json (resume state)")
                    save_progress(config.training.checkpoint_dir, layer_idx,
                                  run_id=run_id, wandb_run_id=wandb_run_id)

            elif needs_training and is_already_done:
                if GLOBAL_RANK == 0:
                    logger.logger.info(
                        f"Skipping layer {current_layer} (already completed, replaying activations)")
                if model.is_moe:
                    model.get_mlp_input_all(train_all)
                    model.get_mlp_input_all(val_all)
                    model.get_mlp_output_all(train_all)
                    model.get_mlp_output_all(val_all)
                else:
                    model.get_layer_activations(train_all)
                    model.get_layer_activations(val_all)
            else:
                model.get_layer_activations(train_all)
                model.get_layer_activations(val_all)
            if GLOBAL_RANK == 0:
                report_pipeline("Refreshing GPTQ calibration activation buffer through layer")
            model.get_layer_activations(gpt_all)

            if GLOBAL_RANK == 0:
                report_pipeline(f"Offloading layer to meta ({current_layer})")
            model.offload_to_meta(current_layer)

            if max_layers is not None and count >= max_layers:
                if GLOBAL_RANK == 0:
                    logger.logger.info(
                        f"Reached --max-layers={max_layers}; stopping early (smoke test)")
                break

            current_layer = model.move_to_next_layer()

            if current_layer:
                if GLOBAL_RANK == 0:
                    logger.logger.info(f"Moving to next layer: {current_layer}")
                next_layer_idx = count
                next_count = count + 1
                next_was_trained = (next_layer_idx <= resume_from_layer
                                    and next_count > first_k_dense_replace
                                    and next_count > config.quantization.start_layer)
                if next_was_trained:
                    model.load_from_disc(current_layer)
                elif count >= config.quantization.start_layer:
                    model.move_layer_to_gpu(current_layer)
                else:
                    model.load_from_disc(current_layer)
            else:
                if GLOBAL_RANK == 0:
                    logger.logger.info("Finished training all layers")
    finally:
        _cleanup_act_cache_mmap(train_all, val_all, gpt_all)

def get_model_wrapper(model_name, tokenizer, batch_size, seqlen, device, dtype, world_size, dummy=False):
    name_lower = model_name.lower()
    if 'llama' in name_lower:
        from src.models.llama import LLaMAWrapper
        return LLaMAWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
    elif 'qwen3.5' in name_lower or 'qwen3.6' in name_lower:
        from transformers import AutoConfig as _AC
        _cfg = _AC.from_pretrained(model_name, trust_remote_code=True)
        _tc = getattr(_cfg, 'text_config', _cfg)
        _is_moe = hasattr(_tc, 'num_experts') or hasattr(_tc, 'num_local_experts')
        if _is_moe and world_size > 1:
            from src.models.qwen35_moe_dist import Qwen35MoeDistributedWrapper
            return Qwen35MoeDistributedWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
        elif _is_moe:
            from src.models.qwen35_moe import Qwen35MoeWrapper
            return Qwen35MoeWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
        else:
            from src.models.qwen35 import Qwen35Wrapper
            return Qwen35Wrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
    elif 'qwen3' in name_lower:
        from transformers import AutoConfig as _AC
        _cfg = _AC.from_pretrained(model_name, trust_remote_code=True)
        _tc = getattr(_cfg, 'text_config', _cfg)
        _is_moe = hasattr(_tc, 'num_experts') or hasattr(_tc, 'num_local_experts')
        if _is_moe and world_size > 1:
            from src.models.qwen3_moe_dist import Qwen3MoeDistributedWrapper
            return Qwen3MoeDistributedWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
        elif _is_moe:
            from src.models.qwen3_moe import Qwen3MoeWrapper
            return Qwen3MoeWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
        else:
            from src.models.qwen3 import Qwen3Wrapper
            return Qwen3Wrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
    elif ('k2.5' in name_lower or 'k2_5' in name_lower) and world_size > 1:
        from src.models.kimi_k25_dist import KimiK25DistributedWrapper
        return KimiK25DistributedWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=dummy)
    elif 'k2.5' in name_lower or 'k2_5' in name_lower:
        from src.models.kimi_k25 import KimiK25Wrapper
        return KimiK25Wrapper(model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=dummy)
    elif 'kimi' in name_lower and world_size > 1:
        from src.models.kimi_k2_dist import KimiK2DistributedWrapper
        return KimiK2DistributedWrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
    elif 'kimi' in name_lower:
        from src.models.kimi_k2 import KimiK2Wrapper
        return KimiK2Wrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
    elif 'gemma-4-31b' in name_lower:
        from src.models.gemma4 import Gemma4Wrapper
        return Gemma4Wrapper(model_name, tokenizer, batch_size, seqlen, device, dtype)
    else:
        raise ValueError(f"Unsupported model family: {model_name}")

def main():
    load_dotenv()
    global GLOBAL_RANK
    args = parse_args()
    config = load_config(args.config)
    progress_reporter.init(config)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=config.distributed.timeout_hours))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        GLOBAL_RANK = dist.get_rank()
        device_count = torch.cuda.device_count()
        device_id = min(local_rank, device_count - 1) if device_count else 0
        torch.cuda.set_device(device_id)

    config_dict = yaml.safe_load(open(args.config, 'r'))

    resume_from_layer = -1
    saved_wandb_id = None

    if GLOBAL_RANK == 0:
        if args.resume is not None:
            if args.resume == 'latest':
                run_id = find_latest_run(config.training.checkpoint_dir)
            else:
                run_id = args.resume
            if run_id is not None:
                run_dir = _run_dir(config.training.checkpoint_dir, run_id)
                progress = load_progress(run_dir)
                if progress is not None:
                    resume_from_layer = progress["last_completed_layer"]
                    saved_wandb_id = progress.get("wandb_run_id")
                    print(f"Resuming run '{run_id}': skipping layers 0..{resume_from_layer}")
                else:
                    print(f"Run '{run_id}' has no progress file; starting fresh")
                    run_id = generate_run_id()
            else:
                print("--resume specified but no runs found; starting fresh")
                run_id = generate_run_id()
        else:
            run_id = generate_run_id()
    else:
        run_id = None

    if world_size > 1:
        id_list = [run_id] if GLOBAL_RANK == 0 else [None]
        dist.broadcast_object_list(id_list, src=0)
        run_id = id_list[0]
        layer_list = [resume_from_layer] if GLOBAL_RANK == 0 else [None]
        dist.broadcast_object_list(layer_list, src=0)
        resume_from_layer = layer_list[0]

    run_dir = _run_dir(config.training.checkpoint_dir, run_id)
    config.training.checkpoint_dir = run_dir
    if GLOBAL_RANK == 0:
        os.makedirs(run_dir, exist_ok=True)
        print(f"Run ID: {run_id}")
        print(f"Checkpoints: {run_dir}")

    if config.wandb.enabled and GLOBAL_RANK == 0:
        model_short = config.model.name.split("/")[-1]
        temp = config.quantization.temperature
        scale = config.quantization.scale
        gsq_bits = config.quantization.gsq_bits
        init_tag = config.quantization.init_method
        gsq_tag = "gsq" if config.quantization.gsq_enabled else "nogsq"
        run_name = (
            f"{model_short}_"
            f"{init_tag}+{gsq_tag}_"
            f"bits={gsq_bits}_"
            f"gsz={config.quantization.groupsize}_"
            f"lg_dtype={config.quantization.logits_dtype}_"
            f"ep={config.training.num_epochs}_"
            f"tmp={temp[0]}-{temp[1]}_scl={scale[0]}-{scale[1]}_"
            f"str={config.quantization.strength}_"
            f"ncal={config.data.num_samples}_"
            f"bsz={config.data.batch_size}_"
            f"lr={config.training.lr1}"
        )
        wandb_kwargs = dict(
            project=config.wandb.project,
            name=run_name,
            config=config_dict,
        )
        if config.wandb.entity:
            wandb_kwargs["entity"] = config.wandb.entity
        if saved_wandb_id is not None:
            wandb_kwargs["id"] = saved_wandb_id
            wandb_kwargs["resume"] = "must"
        wandb.init(**wandb_kwargs)
        wandb.config.update({
            "checkpoint_run_id": run_id,
            "world_size": world_size,
            "num_nodes": max(1, world_size // 4),
            "gpus_per_node": min(world_size, 4),
            "gsq_bits": config.quantization.gsq_bits,
            "gptq_wbits": config.gptq.wbits,
            "init_method": config.quantization.init_method,
            "gsq_enabled": config.quantization.gsq_enabled,
            "logits_dtype": config.quantization.logits_dtype,
        }, allow_val_change=True)
        _code_exclude = {"transformers/", ".venv/", "venv-gsq/", "wandb/", "__pycache__/", ".git/"}
        wandb.run.log_code(
            ".",
            include_fn=lambda path: any(path.endswith(ext) for ext in [".py", ".yaml", ".sh", ".toml"])
                and not any(dir in path for dir in _code_exclude),
        )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True, trust_remote_code=True)
    per_rank_batch = max(1, config.data.batch_size // world_size)
    model = get_model_wrapper(config.model.name, tokenizer, per_rank_batch, config.data.max_length, resolve_device(), config.model.dtype, world_size, dummy=config.model.dummy)
    model.meta_init_std = config.training.meta_init_std
    model.calib_report_divisor = config.logging.calib_report_divisor
    model.batch_report_divisor = config.logging.batch_report_divisor

    logger = QuantizationLogger(config.training.log_dir) if GLOBAL_RANK == 0 else None
    
    train_loader, val_loader, gpt_loader = create_dataloader(
        config.data.dataset_name,
        tokenizer,
        batch_size=per_rank_batch,
        train_samples=config.data.num_samples,
        val_samples=config.data.val_samples,
        gpt_samples=config.gptq.nsamples,
        num_workers=config.data.num_workers,
        max_length=config.data.max_length,
        seed=config.data.seed,
        shuffle_seed=config.data.shuffle_seed,
        shuffle_buffer_size=config.data.shuffle_buffer_size,
        open_thoughts_max_samples=config.data.open_thoughts_max_samples,
    )
    
    try:
        tick = time.time()
        train_all_layers(model, train_loader, val_loader, gpt_loader, logger, config,
                         resume_from_layer=resume_from_layer, run_id=run_id,
                         max_layers=args.max_layers)
        total_time = time.time() - tick
        if GLOBAL_RANK == 0:
            print("Total time:", total_time)
            if config.wandb.enabled:
                wandb.log({"timing/total_wall_clock_sec": total_time})
        
    except KeyboardInterrupt:
        if GLOBAL_RANK == 0:
            logger.logger.info("Training interrupted.")

    finally:
        try:
            del train_loader
            del val_loader
            del gpt_loader
        except NameError:
            pass
        gc.collect()

        if GLOBAL_RANK == 0:
            logger.logger.info("Training finished.")
            report_pipeline("Teardown: closing session (no post-training barriers)")

        # NOTE: We deliberately skip post-training `dist.barrier()` calls.
        # NCCL barriers at end-of-training reliably hang in C++ (uninterruptible
        # by Python), which then traps the elastic agent in `proc.wait()` even
        # though `_distributed_destroy_pg()` has a per-rank timeout. Every rank
        # is about to exit anyway, so synchronization here serves no purpose.

        if GLOBAL_RANK == 0 and config.wandb.enabled:
            try:
                wandb.finish()
            except Exception:
                pass

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

        if GLOBAL_RANK == 0 and dist.is_initialized():
            tout = os.environ.get("GSQ_DIST_DESTROY_TIMEOUT_SEC", "120")
            report_pipeline(
                f"NCCL teardown: destroy_process_group() (join timeout {tout}s; 0=infinite)"
            )

        _distributed_destroy_pg()

        if GLOBAL_RANK == 0:
            report_pipeline("Cleanup complete.")

        # Hard-exit every rank. Returning from main() lets the Python interpreter
        # try to join non-daemon threads (e.g. the destroy_process_group worker
        # thread when it timed out) and run NCCL/CUDA atexit hooks, both of
        # which can deadlock. We've already flushed WandB and freed CUDA caches.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)

if __name__ == "__main__":
    main()
