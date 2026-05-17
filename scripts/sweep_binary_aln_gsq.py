from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from lion_pytorch import Lion
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_binary_ternary_gsq import (  # noqa: E402
    TEXT,
    capture_activations,
    find_module,
    mse_for_weight,
    normalized_mse,
    run_gptq,
    split_rows,
)
from src.quantization import GumbelQuantizer1Bit  # noqa: E402


@dataclass(frozen=True)
class BinaryConfig:
    name: str
    mode: str
    steps: int
    lr1: float
    lr2: float
    temperature: tuple[float, float]
    scale: tuple[float, float]
    weight_decay: float = 0.0


@dataclass(frozen=True)
class FinalPhaseConfig:
    name: str
    warmup: BinaryConfig
    final_mode: str
    final_steps: int
    final_lr1: float
    final_lr2: float
    final_temperature: tuple[float, float]
    final_scale: tuple[float, float]


def build_quantizer(q_init, scales, groupsize, cfg):
    return GumbelQuantizer1Bit(
        q_init.to(torch.bfloat16).clone(),
        scales.float().clone(),
        groupsize,
        std=0.01,
        strength=6.0,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        logits_dtype=torch.float32,
        binary_mode=cfg.mode,
    )


def build_aln_from_standard(standard_quantizer, q_init, groupsize, mode):
    hard, scales = standard_quantizer.get_hard_weights()
    q = GumbelQuantizer1Bit(
        hard.detach().to(torch.bfloat16),
        scales.float().detach().clone(),
        groupsize,
        std=0.01,
        strength=6.0,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        logits_dtype=torch.float32,
        binary_mode=mode,
    )
    with torch.no_grad():
        q.scales.copy_(standard_quantizer.scales.detach())
        if hasattr(standard_quantizer, "sign_logits"):
            signs = torch.where(standard_quantizer.sign_logits.detach() >= 0, 1.0, -1.0)
        else:
            signs = torch.where(q_init >= 0, 1.0, -1.0)
        confidence = torch.ones_like(signs, dtype=torch.float32)
        if hasattr(standard_quantizer, "sign_logits"):
            confidence = standard_quantizer.sign_logits.detach().float().abs().clamp_min(1.0)
        q.sign_scores[0].copy_(torch.where(signs < 0, 1.0 + confidence, torch.ones_like(confidence)))
        q.sign_scores[1].copy_(torch.where(signs > 0, 1.0 + confidence, torch.ones_like(confidence)))
    return q


def optimizer_for(quantizer, cfg):
    params = [p for name, p in quantizer.named_parameters() if name != "scales"]
    return Lion(
        [
            {"params": params, "lr": cfg.lr1, "weight_decay": cfg.weight_decay},
            {"params": [quantizer.scales], "lr": cfg.lr2, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )


def train_quantizer(quantizer, x_train, y_train, x_val, y_val, cfg, seed, eval_points=4):
    torch.manual_seed(seed)
    optimizer = optimizer_for(quantizer, cfg)
    batch_size = min(256, x_train.shape[0])
    best_mse = mse_for_weight(x_val, y_val, quantizer.get_hard_weights()[0], batch_size=128)
    best_step = 0
    eval_interval = max(1, cfg.steps // eval_points)
    losses = []

    start_time = time.time()
    for step in range(cfg.steps):
        t = step / max(1, cfg.steps - 1)
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

        if (step + 1) % eval_interval == 0 or step == cfg.steps - 1:
            hard_mse = mse_for_weight(x_val, y_val, quantizer.get_hard_weights()[0], batch_size=128)
            losses.append({"step": step + 1, "hard_mse": hard_mse})
            if hard_mse < best_mse:
                best_mse = hard_mse
                best_step = step + 1

    final_hard, _ = quantizer.get_hard_weights()
    final_mse = mse_for_weight(x_val, y_val, final_hard, batch_size=128)
    if final_mse < best_mse:
        best_mse = final_mse
        best_step = cfg.steps

    return {
        "initial_hard_mse": losses[0]["hard_mse"] if losses else final_mse,
        "final_hard_mse": final_mse,
        "best_hard_mse": best_mse,
        "best_step": best_step,
        "nonzero_density": float((final_hard != 0).float().mean().item()),
        "seconds": time.time() - start_time,
        "evals": losses,
        "quantizer": quantizer,
    }


def run_single_phase(q_init, scales, x_train, y_train, x_val, y_val, groupsize, cfg, seed):
    quantizer = build_quantizer(q_init, scales, groupsize, cfg)
    result = train_quantizer(quantizer, x_train, y_train, x_val, y_val, cfg, seed)
    result.pop("quantizer")
    return result


def run_final_phase(q_init, scales, x_train, y_train, x_val, y_val, groupsize, cfg, seed):
    standard = build_quantizer(q_init, scales, groupsize, cfg.warmup)
    warmup_result = train_quantizer(standard, x_train, y_train, x_val, y_val, cfg.warmup, seed)
    aln_cfg = BinaryConfig(
        name=cfg.name + "_final",
        mode=cfg.final_mode,
        steps=cfg.final_steps,
        lr1=cfg.final_lr1,
        lr2=cfg.final_lr2,
        temperature=cfg.final_temperature,
        scale=cfg.final_scale,
        weight_decay=0.0,
    )
    aln = build_aln_from_standard(warmup_result["quantizer"], q_init, groupsize, cfg.final_mode)
    final_result = train_quantizer(aln, x_train, y_train, x_val, y_val, aln_cfg, seed + 10_000)
    warmup_result.pop("quantizer")
    final_result.pop("quantizer")
    return {
        "warmup": warmup_result,
        "final": final_result,
        "initial_hard_mse": warmup_result["initial_hard_mse"],
        "warmup_hard_mse": warmup_result["final_hard_mse"],
        "final_hard_mse": final_result["final_hard_mse"],
        "best_hard_mse": min(warmup_result["best_hard_mse"], final_result["best_hard_mse"]),
        "best_step": final_result["best_step"],
        "seconds": warmup_result["seconds"] + final_result["seconds"],
    }


def discovery_configs(steps):
    scheds = [
        ((2.0, 0.2), (10.0, 80.0), "schedA"),
        ((2.0, 0.1), (10.0, 120.0), "schedB"),
        ((1.0, 0.2), (20.0, 120.0), "schedC"),
    ]
    configs = []
    for temp, scale, tag in scheds:
        for lr1 in (3e-4, 6e-4, 1e-3, 1.5e-3, 2e-3):
            for lr2 in (1e-4, 3e-4):
                configs.append(BinaryConfig(f"std_{tag}_lr{lr1:g}_slr{lr2:g}", "standard", steps, lr1, lr2, temp, scale))
    for mode in ("aln", "aln_st"):
        for temp, scale, tag in scheds:
            for lr1 in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3):
                for lr2 in (1e-4, 3e-4):
                    configs.append(BinaryConfig(f"{mode}_{tag}_lr{lr1:g}_slr{lr2:g}", mode, steps, lr1, lr2, temp, scale))
    return configs


def final_phase_configs(top_warmups, final_steps):
    configs = []
    final_scheds = [
        ((2.0, 0.2), (10.0, 80.0), "schedA"),
        ((1.0, 0.2), (10.0, 80.0), "schedB"),
        ((0.5, 0.1), (10.0, 80.0), "schedC"),
    ]
    for warmup in top_warmups:
        for mode in ("aln", "aln_st"):
            for temp, scale, tag in final_scheds:
                for lr1 in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
                    configs.append(
                        FinalPhaseConfig(
                            f"{warmup.name}_then_{mode}_{tag}_lr{lr1:g}",
                            warmup,
                            mode,
                            final_steps,
                            lr1,
                            warmup.lr2,
                            temp,
                            scale,
                        )
                    )
    return configs


def summarize(results, module, kind, limit=5):
    rows = [r for r in results if r["module"] == module and r["kind"] == kind]
    rows.sort(key=lambda r: r["best_hard_mse"])
    return rows[:limit]


def cfg_from_result(result, steps):
    return BinaryConfig(
        result["name"] + "_confirm",
        result["mode"],
        steps,
        result["lr1"],
        result["lr2"],
        tuple(result["temperature"]),
        tuple(result["scale"]),
        result.get("weight_decay", 0.0),
    )


def final_cfg_from_result(result, steps):
    return FinalPhaseConfig(
        result["name"] + "_confirm",
        cfg_from_result(result["warmup_cfg"], result["warmup_cfg"]["steps"]),
        result["final_mode"],
        steps,
        result["final_lr1"],
        result["final_lr2"],
        tuple(result["final_temperature"]),
        tuple(result["final_scale"]),
    )


def result_from_single(module, cfg, result, y_val):
    return {
        "module": module,
        "kind": "single_phase",
        "name": cfg.name,
        "mode": cfg.mode,
        "steps": cfg.steps,
        "lr1": cfg.lr1,
        "lr2": cfg.lr2,
        "weight_decay": cfg.weight_decay,
        "temperature": list(cfg.temperature),
        "scale": list(cfg.scale),
        "initial_hard_mse": result["initial_hard_mse"],
        "final_hard_mse": result["final_hard_mse"],
        "best_hard_mse": result["best_hard_mse"],
        "best_nmse": normalized_mse(result["best_hard_mse"], y_val),
        "final_nmse": normalized_mse(result["final_hard_mse"], y_val),
        "best_step": result["best_step"],
        "seconds": result["seconds"],
    }


def result_from_final(module, cfg, result, y_val):
    warmup_cfg = {
        "name": cfg.warmup.name,
        "mode": cfg.warmup.mode,
        "steps": cfg.warmup.steps,
        "lr1": cfg.warmup.lr1,
        "lr2": cfg.warmup.lr2,
        "weight_decay": cfg.warmup.weight_decay,
        "temperature": list(cfg.warmup.temperature),
        "scale": list(cfg.warmup.scale),
    }
    return {
        "module": module,
        "kind": "final_phase",
        "name": cfg.name,
        "warmup_cfg": warmup_cfg,
        "final_mode": cfg.final_mode,
        "final_steps": cfg.final_steps,
        "final_lr1": cfg.final_lr1,
        "final_lr2": cfg.final_lr2,
        "final_temperature": list(cfg.final_temperature),
        "final_scale": list(cfg.final_scale),
        "initial_hard_mse": result["initial_hard_mse"],
        "warmup_hard_mse": result["warmup_hard_mse"],
        "final_hard_mse": result["final_hard_mse"],
        "best_hard_mse": result["best_hard_mse"],
        "best_nmse": normalized_mse(result["best_hard_mse"], y_val),
        "final_nmse": normalized_mse(result["final_hard_mse"], y_val),
        "seconds": result["seconds"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--modules", nargs="+", default=["model.layers.0.self_attn.k_proj", "model.layers.1.self_attn.k_proj"])
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--train-rows", type=int, default=1024)
    parser.add_argument("--val-rows", type=int, default=384)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--discover-steps", type=int, default=180)
    parser.add_argument("--confirm-steps", type=int, default=300)
    parser.add_argument("--final-steps", type=int, default=120)
    parser.add_argument("--out", default="runtime/gsq/binary_aln_sweep.json")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    captures = capture_activations(model, tokenizer, args.modules, args.max_length)

    output = {
        "model": args.model,
        "modules": args.modules,
        "train_rows": args.train_rows,
        "val_rows": args.val_rows,
        "groupsize": args.groupsize,
        "results": [],
    }

    best_discovery_warmups = None
    best_discovery_finals = None
    for module_idx, module_name in enumerate(args.modules):
        print(f"\n=== {module_name} ===", flush=True)
        layer = find_module(model, module_name)
        x_train, y_train, x_val, y_val = split_rows(captures[module_name]["x"], captures[module_name]["y"], args.train_rows, args.val_rows)
        q_init, scales = run_gptq(layer, x_train, "binary", args.groupsize)
        init_mse = mse_for_weight(x_val, y_val, q_init, batch_size=128)
        output["results"].append(
            {
                "module": module_name,
                "kind": "binary_gptq",
                "name": "binary_gptq",
                "best_hard_mse": init_mse,
                "final_hard_mse": init_mse,
                "best_nmse": normalized_mse(init_mse, y_val),
            }
        )
        print(f"binary GPTQ/init mse={init_mse:.6e}", flush=True)

        if module_idx == 0:
            configs = discovery_configs(args.discover_steps)
        else:
            configs = [cfg_from_result(r, args.confirm_steps) for r in best_discovery_warmups]

        single_results = []
        for cfg_idx, cfg in enumerate(configs):
            result = run_single_phase(q_init, scales, x_train, y_train, x_val, y_val, args.groupsize, cfg, 10_000 + module_idx * 1000 + cfg_idx)
            row = result_from_single(module_name, cfg, result, y_val)
            output["results"].append(row)
            single_results.append(row)
            print(f"{cfg.name} mode={cfg.mode} best={row['best_hard_mse']:.6e} final={row['final_hard_mse']:.6e}", flush=True)

        if module_idx == 0:
            warmups = [cfg_from_result(r, args.discover_steps) for r in sorted(single_results, key=lambda r: r["best_hard_mse"])[:4]]
            final_configs = final_phase_configs(warmups, args.final_steps)
        else:
            final_configs = [final_cfg_from_result(r, args.final_steps) for r in best_discovery_finals]

        final_results = []
        for cfg_idx, cfg in enumerate(final_configs):
            result = run_final_phase(q_init, scales, x_train, y_train, x_val, y_val, args.groupsize, cfg, 20_000 + module_idx * 1000 + cfg_idx)
            row = result_from_final(module_name, cfg, result, y_val)
            output["results"].append(row)
            final_results.append(row)
            print(
                f"{cfg.name} final_mode={cfg.final_mode} warmup={row['warmup_hard_mse']:.6e} "
                f"best={row['best_hard_mse']:.6e} final={row['final_hard_mse']:.6e}",
                flush=True,
            )

        if module_idx == 0:
            best_discovery_warmups = sorted(single_results, key=lambda r: r["best_hard_mse"])[:8]
            best_discovery_finals = sorted(final_results, key=lambda r: r["best_hard_mse"])[:8]

        del x_train, y_train, x_val, y_val, q_init, scales
        torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
