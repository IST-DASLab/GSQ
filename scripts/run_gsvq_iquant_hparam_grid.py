#!/usr/bin/env python3
"""Broad hyperparameter grid for fixed-scale GSVQ over GGUF IQuant tensors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = REPO_ROOT / "scripts" / "run_gsvq_iquant_sweep.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--out", default="runtime/gsvq_iquant_hparam")
    parser.add_argument("--devices", default="cuda:3,cuda:5,cuda:6,cuda:7")
    parser.add_argument("--search-layers", default="0,7,14,35")
    parser.add_argument("--search-max-vectors", type=int, default=8192)
    parser.add_argument("--search-chunk-vectors", type=int, default=8192)
    parser.add_argument("--full-chunk-vectors", type=int, default=131072)
    parser.add_argument("--top-k-full", type=int, default=6)
    parser.add_argument("--jobs-per-stage", type=int, default=0,
                        help="0 means one worker per listed device.")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--limit-configs", type=int, default=0)
    return parser.parse_args()


def grid_configs():
    configs = []

    def add(name, **kwargs):
        cfg = {
            "id": name,
            "preset": "tuned",
            "steps": 20,
            "lr": 0.05,
            "weight_decay": 0.0,
            "temp_start": 1.5,
            "temp_end": 0.1,
            "scale_start": 1.0,
            "scale_end": 40.0,
            "candidate_count": 8,
            "neighbor_candidates": 4,
            "target_candidates": 4,
            "std": 0.01,
            "strength": 0.25,
            "optimizer_name": "adamw",
            "grad_clip": 0.0,
            "restarts": 1,
            "rotation_trick": False,
        }
        cfg.update(kwargs)
        configs.append(cfg)

    add("original", preset="original")
    add("current_tuned")

    for lr in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        add(f"lr_{lr:g}", lr=lr)

    for steps in [5, 10, 20, 50, 100]:
        add(f"steps_{steps}", steps=steps)

    for cand, neigh, target in [
        (4, 2, 2),
        (8, 4, 4),
        (8, 8, 0),
        (8, 0, 8),
        (16, 8, 8),
        (16, 4, 12),
        (16, 12, 4),
        (32, 16, 16),
    ]:
        add(f"cand_{cand}_n{neigh}_t{target}",
            candidate_count=cand, neighbor_candidates=neigh, target_candidates=target)

    for strength in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]:
        add(f"strength_{strength:g}", strength=strength)

    for std in [0.0, 0.005, 0.01, 0.03, 0.05, 0.1]:
        add(f"std_{std:g}", std=std)

    for temp_start, temp_end in [(2.0, 0.05), (1.5, 0.05), (1.0, 0.05), (0.7, 0.05), (1.5, 0.2), (0.7, 0.2)]:
        add(f"temp_{temp_start:g}_{temp_end:g}", temp_start=temp_start, temp_end=temp_end)

    for scale_start, scale_end in [(0.1, 10.0), (0.5, 20.0), (1.0, 40.0), (1.0, 100.0), (5.0, 100.0), (10.0, 200.0)]:
        add(f"scale_{scale_start:g}_{scale_end:g}", scale_start=scale_start, scale_end=scale_end)

    for opt in ["adam", "adamw", "lion", "sgd"]:
        add(f"opt_{opt}", optimizer_name=opt)

    for clip in [0.1, 1.0, 5.0]:
        add(f"clip_{clip:g}", grad_clip=clip)

    add("restart_2", restarts=2)
    add("restart_3", restarts=3)
    add("rot_trick", rotation_trick=True)

    # Combined configs that intentionally push the search harder than the
    # one-factor grid.
    add("hard_lr01_steps50_c8", lr=0.1, steps=50)
    add("hard_lr02_steps50_c8", lr=0.2, steps=50)
    add("hard_lr01_steps100_c8", lr=0.1, steps=100)
    add("hard_lr02_steps100_c8", lr=0.2, steps=100)
    add("hard_lr05_steps50_c4", lr=0.5, steps=50, candidate_count=4, neighbor_candidates=2, target_candidates=2)
    add("hard_lr01_steps50_c16", lr=0.1, steps=50, candidate_count=16, neighbor_candidates=8, target_candidates=8)
    add("hard_lr02_steps50_c16", lr=0.2, steps=50, candidate_count=16, neighbor_candidates=8, target_candidates=8)
    add("hard_lr01_strength0", lr=0.1, strength=0.0, std=0.03)
    add("hard_lr02_strength0", lr=0.2, strength=0.0, std=0.03)
    add("hard_c8_target_only_lr01", lr=0.1, candidate_count=8, neighbor_candidates=0, target_candidates=8)
    add("hard_c8_target_only_lr02", lr=0.2, candidate_count=8, neighbor_candidates=0, target_candidates=8)
    add("hard_c8_neighbor_only_lr01", lr=0.1, candidate_count=8, neighbor_candidates=8, target_candidates=0)
    add("hard_scale100_lr01", lr=0.1, scale_start=1.0, scale_end=100.0)
    add("hard_scale200_lr01", lr=0.1, scale_start=5.0, scale_end=200.0)
    add("hard_temp_low_lr01", lr=0.1, temp_start=0.7, temp_end=0.05)
    add("hard_lion_lr001_steps50", optimizer_name="lion", lr=0.001, steps=50)
    add("hard_lion_lr005_steps50", optimizer_name="lion", lr=0.005, steps=50)
    add("hard_restart3_lr01", lr=0.1, restarts=3)
    return configs


def run_cmd_for_config(args, cfg, device, stage, out_root):
    run_out = out_root / stage / cfg["id"]
    run_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SWEEP_SCRIPT),
        "--gguf", args.gguf,
        "--hf-model", args.hf_model,
        "--out", str(run_out),
        "--preset", cfg["preset"],
        "--device", device,
        "--seed", "0",
    ]
    if stage == "search":
        cmd += [
            "--layers", args.search_layers,
            "--max-vectors-per-tensor", str(args.search_max_vectors),
            "--chunk-vectors", str(args.search_chunk_vectors),
        ]
    else:
        cmd += ["--chunk-vectors", str(args.full_chunk_vectors)]

    if cfg["preset"] != "original":
        cmd += [
            "--steps", str(cfg["steps"]),
            "--lr", str(cfg["lr"]),
            "--weight-decay", str(cfg["weight_decay"]),
            "--temp-start", str(cfg["temp_start"]),
            "--temp-end", str(cfg["temp_end"]),
            "--scale-start", str(cfg["scale_start"]),
            "--scale-end", str(cfg["scale_end"]),
            "--candidate-count", str(cfg["candidate_count"]),
            "--neighbor-candidates", str(cfg["neighbor_candidates"]),
            "--target-candidates", str(cfg["target_candidates"]),
            "--std", str(cfg["std"]),
            "--strength", str(cfg["strength"]),
            "--optimizer-name", cfg["optimizer_name"],
            "--grad-clip", str(cfg["grad_clip"]),
            "--restarts", str(cfg["restarts"]),
        ]
        if cfg["rotation_trick"]:
            cmd.append("--rotation-trick")

    log_path = run_out / "subprocess.log"
    started = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    result = {
        "id": cfg["id"],
        "stage": stage,
        "device": device,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "out": str(run_out),
        "log": str(log_path),
        **{k: v for k, v in cfg.items() if k != "id"},
    }
    if proc.returncode == 0:
        result.update(read_overall(run_out))
    return result


def read_overall(run_out):
    csv_paths = sorted(run_out.glob("*/tensor_results.csv"))
    if not csv_paths:
        return {}
    rows = list(csv.DictReader(open(csv_paths[-1])))
    elems = 0
    init = 0.0
    accepted = 0.0
    raw = 0.0
    improved = 0
    chunks = 0
    for row in rows:
        e = int(row["vectors"]) * int(row["vector_dim"])
        elems += e
        init += float(row["initial_mse"]) * e
        accepted += float(row["accepted_mse"]) * e
        raw += float(row["raw_best_mse"]) * e
        improved += int(row["improved_chunks"])
        chunks += int(row["chunks"])
    init /= elems
    accepted /= elems
    raw /= elems
    return {
        "result_dir": str(csv_paths[-1].parent),
        "tensors": len(rows),
        "elements": elems,
        "initial_mse": init,
        "accepted_mse": accepted,
        "raw_best_mse": raw,
        "accepted_abs_delta": init - accepted,
        "accepted_rel_delta": (init - accepted) / max(init, 1e-30),
        "raw_abs_delta": init - raw,
        "raw_rel_delta": (init - raw) / max(init, 1e-30),
        "improved_chunks": improved,
        "chunks": chunks,
    }


def write_rows(path, rows):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_stage(args, configs, devices, stage, out_root):
    groups = [[] for _ in devices]
    for i, cfg in enumerate(configs):
        groups[i % len(devices)].append(cfg)

    def worker(device, group):
        out = []
        for cfg in group:
            print(f"{stage}: starting {cfg['id']} on {device}", flush=True)
            result = run_cmd_for_config(args, cfg, device, stage, out_root)
            rel = result.get("accepted_rel_delta")
            rel_s = "n/a" if rel is None else f"{100 * rel:.4f}%"
            print(
                f"{stage}: finished {cfg['id']} rc={result['returncode']} rel={rel_s} "
                f"time={result['elapsed_sec']:.1f}s",
                flush=True,
            )
            out.append(result)
        return out

    rows = []
    with ThreadPoolExecutor(max_workers=min(len(devices), args.jobs_per_stage or len(devices))) as pool:
        futures = [pool.submit(worker, device, group) for device, group in zip(devices, groups) if group]
        for fut in as_completed(futures):
            rows.extend(fut.result())
    rows.sort(key=lambda r: r.get("accepted_rel_delta", -1), reverse=True)
    write_rows(out_root / f"{stage}_summary.csv", rows)
    with open(out_root / f"{stage}_summary.json", "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    return rows


def main():
    args = parse_args()
    out_root = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    configs = grid_configs()
    if args.limit_configs > 0:
        configs = configs[:args.limit_configs]
    with open(out_root / "grid.json", "w") as f:
        json.dump({"args": vars(args), "configs": configs}, f, indent=2, sort_keys=True)

    search_rows = []
    if not args.skip_search:
        search_rows = run_stage(args, configs, devices, "search", out_root)
        print("\nTop search configs:", flush=True)
        for row in search_rows[: min(10, len(search_rows))]:
            print(
                f"{row['id']}: rel={100 * row.get('accepted_rel_delta', 0):.4f}% "
                f"delta={row.get('accepted_abs_delta', 0):.8e}",
                flush=True,
            )

    if not args.skip_full:
        if search_rows:
            full_configs = []
            seen = set()
            for row in search_rows:
                if row.get("returncode") != 0:
                    continue
                cfg = next(c for c in configs if c["id"] == row["id"])
                if cfg["id"] not in seen:
                    full_configs.append(cfg)
                    seen.add(cfg["id"])
                if len(full_configs) >= args.top_k_full:
                    break
        else:
            full_configs = configs[: args.top_k_full]
        full_rows = run_stage(args, full_configs, devices, "full", out_root)
        print("\nTop full configs:", flush=True)
        for row in full_rows:
            print(
                f"{row['id']}: rel={100 * row.get('accepted_rel_delta', 0):.4f}% "
                f"delta={row.get('accepted_abs_delta', 0):.8e} result={row.get('result_dir')}",
                flush=True,
            )

    print(f"\nWrote hparam sweep to {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
