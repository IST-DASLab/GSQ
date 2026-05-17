"""Small real-layer reconstruction benchmark for binary and ternary GSQ.

The script captures activations from selected linear layers, runs GPTQ
initialization, and compares held-out hard-weight reconstruction error for
standard GSQ and the new ALN ablations.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
from lion_pytorch import Lion
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.prior.gptq import GPTQ
from src.prior.quant import Quantizer
from src.quantization import GumbelQuantizer1Bit, GumbelQuantizerTernary


TEXT = """
Gumbel Softmax Quantization optimizes discrete scalar assignments by matching
full precision module outputs on calibration data. A useful small benchmark
should therefore measure held out reconstruction error after hardening, not only
the relaxed training objective.

Large language model layers differ in dynamic range, activation geometry, and
the sensitivity of each output channel. Projection matrices in early transformer
blocks are a compact way to test whether a quantization method is stable before
spending full calibration time on the complete model.

Binary quantization has no zero codepoint, so every coordinate must decide a
sign and rely heavily on the group scale. Ternary quantization adds an explicit
zero mask, which can reduce noise, but imposing a fixed density may help or hurt
depending on whether the selected nonzero fraction matches the layer.
""" * 64


@dataclass(frozen=True)
class TrainConfig:
    name: str
    steps: int
    lr1: float
    lr2: float
    weight_decay: float
    temperature: tuple[float, float]
    scale: tuple[float, float]
    density: float | None = None
    scope: str | None = None
    binary_mode: str = "standard"


def build_config(gsq_bits, groupsize, trits=False):
    return SimpleNamespace(
        quantization=SimpleNamespace(
            gsq_bits=gsq_bits,
            std=0.01,
            strength=6.0,
            temperature=[2.0, 0.05],
            scale=[10.0, 80.0],
            ternary_mask_mode="standard",
            ternary_density=0.5,
            ternary_density_scope="row",
            ternary_density_eps=1e-6,
        ),
        gptq=SimpleNamespace(
            wbits=1 if gsq_bits == 1 else 2,
            sym=True,
            trits=trits,
            percdamp=0.01,
            blocksize=128,
            groupsize=groupsize,
            static_groups=False,
            prunen=0,
            prunem=0,
        ),
        training=SimpleNamespace(lr1=1e-4, lr2=1e-4, weight_decay=0.0, lion_betas=[0.9, 0.95]),
    )


def find_module(model, name):
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Module {name!r} not found")
    return modules[name]


def capture_activations(model, tokenizer, module_names, max_length):
    captured = {name: {} for name in module_names}
    handles = []

    def hook_for(name):
        def hook(_module, inputs, output):
            captured[name]["x"] = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]).cpu()
            captured[name]["y"] = output.detach().float().reshape(-1, output.shape[-1]).cpu()

        return hook

    for name in module_names:
        handles.append(find_module(model, name).register_forward_hook(hook_for(name)))

    inputs = tokenizer(TEXT, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        model(**inputs, use_cache=False)

    for handle in handles:
        handle.remove()

    return captured


def split_rows(x, y, train_rows, val_rows):
    if x.shape[0] < train_rows + val_rows:
        raise ValueError(f"Need {train_rows + val_rows} captured rows, got {x.shape[0]}")
    return (
        x[:train_rows].cuda(non_blocking=True),
        y[:train_rows].cuda(non_blocking=True),
        x[train_rows : train_rows + val_rows].cuda(non_blocking=True),
        y[train_rows : train_rows + val_rows].cuda(non_blocking=True),
    )


def mse_for_weight(x, y, weight, batch_size):
    weight = weight.float()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].float()
            yb = y[start : start + batch_size].float()
            pred = xb @ weight.t()
            loss = (pred - yb).pow(2).sum().item()
            total += loss
            count += yb.numel()
    return total / count


def normalized_mse(mse, y):
    denom = y.float().pow(2).mean().item()
    return mse / max(denom, 1e-12)


def run_gptq(layer, x_train, kind, groupsize):
    if kind == "ternary":
        return run_compatible_ternary_gptq(layer, x_train, groupsize)

    cfg = build_config(1 if kind == "binary" else 2, groupsize, trits=(kind == "ternary"))
    gptq = GPTQ(layer, "bench_linear", cfg, torch.device("cuda"), torch.float32)
    gptq.quantizer = Quantizer()
    gptq.quantizer.configure(
        cfg.gptq.wbits,
        perchannel=True,
        sym=True,
        mse=True,
        trits=cfg.gptq.trits,
    )
    for start in range(0, x_train.shape[0], 128):
        gptq.add_batch(x_train[start : start + 128], None)
    q, scales = gptq.fasterquant(
        logging=None,
        blocksize=cfg.gptq.blocksize,
        percdamp=cfg.gptq.percdamp,
        groupsize=groupsize,
        static_groups=False,
    )
    gptq.free()
    return q.detach(), scales.detach()


def ternary_quantize(x, scale):
    scale = scale.abs().clamp_min(torch.finfo(torch.float32).eps)
    return torch.where(x > scale / 2, scale, torch.where(x < -scale / 2, -scale, torch.zeros_like(x)))


def find_ternary_scales(w_group, grid=50):
    max_abs = w_group.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).eps)
    best_scale = max_abs.clone()
    best_err = torch.full_like(max_abs, float("inf"))
    for i in range(grid):
        shrink = 1.0 - 0.8 * i / max(1, grid - 1)
        scale = max_abs * shrink
        q = ternary_quantize(w_group, scale.unsqueeze(1))
        err = (q - w_group).pow(2).sum(dim=1)
        use = err < best_err
        best_err = torch.where(use, err, best_err)
        best_scale = torch.where(use, scale, best_scale)
    return best_scale


def run_compatible_ternary_gptq(layer, x_train, groupsize):
    # The repository's legacy trits path can produce magnitudes that are not
    # directly representable by the ternary GSQ quantizer. This local GPTQ pass
    # keeps the initializer on the shared {-s, 0, +s} grid.
    cfg = build_config(2, groupsize, trits=True)
    gptq = GPTQ(layer, "bench_linear", cfg, torch.device("cuda"), torch.float32)
    for start in range(0, x_train.shape[0], 128):
        gptq.add_batch(x_train[start : start + 128], None)
    gptq.cholesky(cfg.gptq.percdamp)

    w = layer.weight.detach().float().clone()
    w[:, gptq.dead] = 0
    rows, columns = w.shape
    n_groups = (columns + groupsize - 1) // groupsize
    group_scales = torch.zeros(rows, n_groups, device=w.device, dtype=torch.float32)

    h = torch.cholesky_inverse(gptq.H.float())
    hinv = torch.linalg.cholesky(h, upper=True)

    for i1 in range(0, columns, cfg.gptq.blocksize):
        i2 = min(i1 + cfg.gptq.blocksize, columns)
        count = i2 - i1
        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]

        for i in range(count):
            col = i1 + i
            if col % groupsize == 0:
                g = col // groupsize
                g_end = min(col + groupsize, columns)
                group_scales[:, g] = find_ternary_scales(w[:, col:g_end])

            g = col // groupsize
            d = hinv1[i, i]
            q = ternary_quantize(w1[:, i], group_scales[:, g])
            q1[:, i] = q
            err = (w1[:, i] - q) / d
            w1[:, i:] -= err.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
            err1[:, i] = err

        w[:, i1:i2] = q1
        w[:, i2:] -= err1.matmul(hinv[i1:i2, i2:])

    q = w.clone()
    gptq.free()
    return q.detach(), group_scales.detach()


def optimizer_for(quantizer, cfg):
    if isinstance(quantizer, GumbelQuantizerTernary) and quantizer.mask_mode == "fixed_density":
        return Lion(
            [
                {"params": [quantizer.sign_logits], "lr": cfg.lr1, "weight_decay": cfg.weight_decay},
                {"params": [quantizer.mask_logits], "lr": cfg.lr1, "weight_decay": 0.0},
                {"params": [quantizer.scales], "lr": cfg.lr2, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.95),
        )
    logit_params = [p for n, p in quantizer.named_parameters() if n != "scales"]
    return Lion(
        [
            {"params": logit_params, "lr": cfg.lr1, "weight_decay": cfg.weight_decay},
            {"params": [quantizer.scales], "lr": cfg.lr2, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )


def build_quantizer(kind, q_init, scales, groupsize, cfg):
    kwargs = dict(
        Q=q_init.to(torch.bfloat16).clone(),
        scales=scales.float().clone(),
        groupsize=groupsize,
        std=0.01,
        strength=6.0,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        logits_dtype=torch.float32,
    )
    if kind == "binary":
        return GumbelQuantizer1Bit(**kwargs, binary_mode=cfg.binary_mode)
    if kind == "ternary_standard":
        return GumbelQuantizerTernary(**kwargs)
    if kind == "ternary_fixed":
        return GumbelQuantizerTernary(
            **kwargs,
            mask_mode="fixed_density",
            density=cfg.density,
            density_scope=cfg.scope,
            density_eps=1e-6,
        )
    raise ValueError(kind)


def train_gsq(kind, q_init, scales, x_train, y_train, x_val, y_val, groupsize, cfg, seed):
    torch.manual_seed(seed)
    quantizer = build_quantizer(kind, q_init, scales, groupsize, cfg)
    optimizer = optimizer_for(quantizer, cfg)

    batch_size = min(256, x_train.shape[0])
    best = {"hard_mse": float("inf"), "step": -1}
    initial_hard, _ = quantizer.get_hard_weights()
    initial_hard_mse = mse_for_weight(x_val, y_val, initial_hard, batch_size=128)

    start_time = time.time()
    for step in range(cfg.steps):
        denom = max(1, cfg.steps - 1)
        t = step / denom
        temperature = cfg.temperature[0] + (cfg.temperature[1] - cfg.temperature[0]) * t
        scale = cfg.scale[0] + (cfg.scale[1] - cfg.scale[0]) * t
        idx = torch.randint(0, x_train.shape[0], (batch_size,), device=x_train.device)
        xb = x_train[idx].to(torch.bfloat16)
        yb = y_train[idx].float()

        optimizer.zero_grad(set_to_none=True)
        wq = quantizer.forward(temperature, scale)
        pred = xb @ wq.t()
        loss = (pred.float() - yb).pow(2).mean()
        loss.backward()
        optimizer.step()

        if step == cfg.steps - 1 or (step + 1) % max(10, cfg.steps // 4) == 0:
            hard, _ = quantizer.get_hard_weights()
            hard_mse = mse_for_weight(x_val, y_val, hard, batch_size=128)
            if hard_mse < best["hard_mse"]:
                best = {"hard_mse": hard_mse, "step": step + 1}

    hard, _ = quantizer.get_hard_weights()
    final_hard_mse = mse_for_weight(x_val, y_val, hard, batch_size=128)
    nonzero_density = float((hard != 0).float().mean().item())
    scale_mean = float(quantizer.scales.detach().abs().mean().item())
    elapsed = time.time() - start_time

    return {
        "name": cfg.name,
        "kind": kind,
        "steps": cfg.steps,
        "lr1": cfg.lr1,
        "lr2": cfg.lr2,
        "weight_decay": cfg.weight_decay,
        "temperature": list(cfg.temperature),
        "scale": list(cfg.scale),
        "density": cfg.density,
        "scope": cfg.scope,
        "binary_mode": cfg.binary_mode,
        "initial_hard_mse": initial_hard_mse,
        "final_hard_mse": final_hard_mse,
        "best_hard_mse": best["hard_mse"],
        "best_step": best["step"],
        "nonzero_density": nonzero_density,
        "scale_mean": scale_mean,
        "seconds": elapsed,
    }


def configs_for_stage(stage_steps):
    sched_a = ((2.0, 0.2), (10.0, 80.0))
    sched_b = ((2.0, 0.5), (5.0, 40.0))
    binary = [
        TrainConfig("bin_lr1e-4_wd0_schedA", stage_steps, 1e-4, 1e-4, 0.0, *sched_a),
        TrainConfig("bin_lr3e-4_wd0_schedA", stage_steps, 3e-4, 1e-4, 0.0, *sched_a),
        TrainConfig("bin_lr1e-4_wd0.1_schedA", stage_steps, 1e-4, 1e-4, 0.1, *sched_a),
        TrainConfig("bin_lr1e-4_wd0_schedB", stage_steps, 1e-4, 1e-4, 0.0, *sched_b),
        TrainConfig("bin_aln_lr1e-4", stage_steps, 1e-4, 1e-4, 0.0, *sched_a, binary_mode="aln"),
        TrainConfig("bin_aln_st_lr1e-4", stage_steps, 1e-4, 1e-4, 0.0, *sched_a, binary_mode="aln_st"),
        TrainConfig("bin_aln_st_lr3e-4", stage_steps, 3e-4, 1e-4, 0.0, *sched_a, binary_mode="aln_st"),
        TrainConfig("bin_aln_st_lr1e-3", stage_steps, 1e-3, 1e-4, 0.0, *sched_a, binary_mode="aln_st"),
    ]
    ternary_standard = [
        TrainConfig("tern_std_lr1e-5_wd0", stage_steps, 1e-5, 1e-5, 0.0, *sched_a),
        TrainConfig("tern_std_lr3e-5_wd0", stage_steps, 3e-5, 3e-5, 0.0, *sched_a),
        TrainConfig("tern_std_lr1e-5_wd0.01", stage_steps, 1e-5, 1e-5, 0.01, *sched_a),
        TrainConfig("tern_std_lr1e-4_wd0.1", stage_steps, 1e-4, 1e-4, 0.1, *sched_a),
        TrainConfig("tern_std_lr3e-4_wd0.1", stage_steps, 3e-4, 1e-4, 0.1, *sched_a),
        TrainConfig("tern_std_lr1e-4_wd0", stage_steps, 1e-4, 1e-4, 0.0, *sched_a),
        TrainConfig("tern_std_lr1e-4_wd0.1_schedB", stage_steps, 1e-4, 1e-4, 0.1, *sched_b),
    ]
    ternary_fixed = []
    for density in (0.25, 0.3125, 0.34375, 0.375, 0.5):
        for scope in ("row", "group"):
            ternary_fixed.append(
                TrainConfig(
                    f"tern_fixed_d{density:g}_{scope}_lr3e-5",
                    stage_steps,
                    3e-5,
                    3e-5,
                    0.0,
                    *sched_a,
                    density=density,
                    scope=scope,
                )
            )
    ternary_fixed.extend(
        [
            TrainConfig("tern_fixed_d0.34375_row_lr1e-5", stage_steps, 1e-5, 1e-5, 0.0, *sched_a, density=0.34375, scope="row"),
            TrainConfig("tern_fixed_d0.34375_group_lr1e-5", stage_steps, 1e-5, 1e-5, 0.0, *sched_a, density=0.34375, scope="group"),
            TrainConfig("tern_fixed_d0.5_row_lr3e-4", stage_steps, 3e-4, 1e-4, 0.0, *sched_a, density=0.5, scope="row"),
            TrainConfig("tern_fixed_d0.5_group_lr3e-4", stage_steps, 3e-4, 1e-4, 0.0, *sched_a, density=0.5, scope="group"),
            TrainConfig("tern_fixed_d0.5_row_schedB", stage_steps, 1e-4, 1e-4, 0.0, *sched_b, density=0.5, scope="row"),
            TrainConfig("tern_fixed_d0.5_group_schedB", stage_steps, 1e-4, 1e-4, 0.0, *sched_b, density=0.5, scope="group"),
        ]
    )
    return binary, ternary_standard, ternary_fixed


def top_configs(results, kind, n):
    candidates = [r for r in results if r.get("kind") == kind]
    candidates.sort(key=lambda r: r["best_hard_mse"])
    return candidates[:n]


def cfg_from_result(result, steps):
    return TrainConfig(
        result["name"] + "_confirm",
        steps,
        result["lr1"],
        result["lr2"],
        result["weight_decay"],
        tuple(result["temperature"]),
        tuple(result["scale"]),
        density=result["density"],
        scope=result["scope"],
        binary_mode=result.get("binary_mode", "standard"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["model.layers.0.self_attn.k_proj", "model.layers.1.self_attn.k_proj"],
    )
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--train-rows", type=int, default=1024)
    parser.add_argument("--val-rows", type=int, default=384)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--sweep-steps", type=int, default=80)
    parser.add_argument("--confirm-steps", type=int, default=120)
    parser.add_argument("--out", default="runtime/gsq/bench_binary_ternary_layers.json")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(2026)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    captures = capture_activations(model, tokenizer, args.modules, args.max_length)
    binary_cfgs, tern_std_cfgs, tern_fixed_cfgs = configs_for_stage(args.sweep_steps)

    all_results = {
        "model": args.model,
        "modules": args.modules,
        "train_rows": args.train_rows,
        "val_rows": args.val_rows,
        "groupsize": args.groupsize,
        "results": [],
    }

    for module_idx, module_name in enumerate(args.modules):
        print(f"\n=== {module_name} ===", flush=True)
        module = find_module(model, module_name)
        layer = copy.deepcopy(module).cuda().eval()
        x_train, y_train, x_val, y_val = split_rows(
            captures[module_name]["x"],
            captures[module_name]["y"],
            args.train_rows,
            args.val_rows,
        )
        full_mse = mse_for_weight(x_val, y_val, layer.weight.detach(), batch_size=128)
        print(f"full_precision_mse={full_mse:.6e}", flush=True)

        q_binary, scales_binary = run_gptq(layer, x_train, "binary", args.groupsize)
        binary_gptq_mse = mse_for_weight(x_val, y_val, q_binary, batch_size=128)
        binary_baseline = {
            "module": module_name,
            "kind": "binary_gptq",
            "final_hard_mse": binary_gptq_mse,
            "best_hard_mse": binary_gptq_mse,
            "nmse": normalized_mse(binary_gptq_mse, y_val),
            "nonzero_density": float((q_binary != 0).float().mean().item()),
        }
        all_results["results"].append(binary_baseline)
        print(f"binary GPTQ/init mse={binary_gptq_mse:.6e} nmse={binary_baseline['nmse']:.6e}", flush=True)

        q_ternary, scales_ternary = run_gptq(layer, x_train, "ternary", args.groupsize)
        ternary_gptq_mse = mse_for_weight(x_val, y_val, q_ternary, batch_size=128)
        ternary_baseline = {
            "module": module_name,
            "kind": "ternary_gptq",
            "final_hard_mse": ternary_gptq_mse,
            "best_hard_mse": ternary_gptq_mse,
            "nmse": normalized_mse(ternary_gptq_mse, y_val),
            "nonzero_density": float((q_ternary != 0).float().mean().item()),
        }
        all_results["results"].append(ternary_baseline)
        print(f"ternary GPTQ/init mse={ternary_gptq_mse:.6e} nmse={ternary_baseline['nmse']:.6e}", flush=True)

        if module_idx == 0:
            run_binary_cfgs = binary_cfgs
            run_std_cfgs = tern_std_cfgs
            run_fixed_cfgs = tern_fixed_cfgs
        else:
            previous = [r for r in all_results["results"] if r.get("module") == args.modules[0]]
            run_binary_cfgs = [cfg_from_result(r, args.confirm_steps) for r in top_configs(previous, "binary", 2)]
            run_std_cfgs = [cfg_from_result(r, args.confirm_steps) for r in top_configs(previous, "ternary_standard", 2)]
            run_fixed_cfgs = [cfg_from_result(r, args.confirm_steps) for r in top_configs(previous, "ternary_fixed", 4)]

        for cfg in run_binary_cfgs:
            result = train_gsq("binary", q_binary, scales_binary, x_train, y_train, x_val, y_val, args.groupsize, cfg, 1000 + module_idx)
            result["module"] = module_name
            result["nmse"] = normalized_mse(result["final_hard_mse"], y_val)
            result["best_nmse"] = normalized_mse(result["best_hard_mse"], y_val)
            all_results["results"].append(result)
            print(f"{result['kind']} {result['name']} final={result['final_hard_mse']:.6e} best={result['best_hard_mse']:.6e}", flush=True)

        for cfg in run_std_cfgs:
            result = train_gsq("ternary_standard", q_ternary, scales_ternary, x_train, y_train, x_val, y_val, args.groupsize, cfg, 2000 + module_idx)
            result["module"] = module_name
            result["nmse"] = normalized_mse(result["final_hard_mse"], y_val)
            result["best_nmse"] = normalized_mse(result["best_hard_mse"], y_val)
            all_results["results"].append(result)
            print(f"{result['kind']} {result['name']} final={result['final_hard_mse']:.6e} best={result['best_hard_mse']:.6e}", flush=True)

        for cfg in run_fixed_cfgs:
            result = train_gsq("ternary_fixed", q_ternary, scales_ternary, x_train, y_train, x_val, y_val, args.groupsize, cfg, 3000 + module_idx)
            result["module"] = module_name
            result["nmse"] = normalized_mse(result["final_hard_mse"], y_val)
            result["best_nmse"] = normalized_mse(result["best_hard_mse"], y_val)
            all_results["results"].append(result)
            print(f"{result['kind']} {result['name']} final={result['final_hard_mse']:.6e} best={result['best_hard_mse']:.6e} density={result['nonzero_density']:.3f}", flush=True)

        del layer, x_train, y_train, x_val, y_val
        torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
