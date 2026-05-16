#!/usr/bin/env python3
"""Checkpoint 1 for IQuant GSVQ: hard reconstruction error must decrease."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantization.gsvq import (  # noqa: E402
    FactorizedIQuantGSVQ,
    build_synthetic_iquant_problem,
    format_history,
    train_gsvq_reconstruction,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a fixed-scale IQuant-style GSVQ reconstruction checkpoint."
    )
    parser.add_argument("--qtype", default="IQ2_XS",
                        choices=["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S"],
                        help="IQuant magnitude grid to use for the synthetic checkpoint.")
    parser.add_argument("--vectors", type=int, default=4096,
                        help="Number of small VQ vectors in the synthetic task.")
    parser.add_argument("--steps", type=int, default=200,
                        help="GSVQ optimization steps.")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="AdamW learning rate for assignment logits.")
    parser.add_argument("--candidate-count", type=int, default=16,
                        help="Per-vector candidate count for magnitude/sign assignments.")
    parser.add_argument("--neighbor-candidates", type=int, default=8,
                        help="Candidates taken from codebook-nearest neighbors of the init code.")
    parser.add_argument("--target-candidates", type=int, default=8,
                        help="Candidates taken from target-nearest codebook entries.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rotation-trick", action="store_true",
                        help="Enable gradient-only rotation preconditioning.")
    parser.add_argument("--json", action="store_true",
                        help="Print only machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    problem = build_synthetic_iquant_problem(
        qtype_name=args.qtype,
        num_vectors=args.vectors,
        seed=args.seed,
        noise_std=args.noise_std,
        device=args.device,
    )
    quantizer = FactorizedIQuantGSVQ(
        problem["target_vectors"],
        problem["scales"],
        problem["magnitude_codebook"],
        problem["sign_codebook"],
        problem["init_magnitude_indices"],
        problem["init_sign_indices"],
        importance=problem["importance"],
        candidate_count=args.candidate_count,
        neighbor_candidates=args.neighbor_candidates,
        target_candidates=args.target_candidates,
        rotation_trick=args.rotation_trick,
    ).to(args.device)

    history = train_gsvq_reconstruction(
        quantizer,
        steps=args.steps,
        lr=args.lr,
    )
    decrease = history.initial_hard_mse - history.best_hard_mse
    rel_decrease = decrease / max(history.initial_hard_mse, 1e-30)

    payload = {
        "checkpoint": "gsvq_reconstruction_decrease",
        "qtype": args.qtype,
        "vectors": args.vectors,
        "steps": args.steps,
        "device": str(args.device),
        "rotation_trick": bool(args.rotation_trick),
        "initial_hard_mse": history.initial_hard_mse,
        "best_hard_mse": history.best_hard_mse,
        "final_hard_mse": history.final_hard_mse,
        "absolute_decrease": decrease,
        "relative_decrease": rel_decrease,
        "passed": history.decreased,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("GSVQ reconstruction checkpoint")
        print(json.dumps(payload, indent=2, sort_keys=True))
        print(format_history(history))

    if not history.decreased:
        print(
            "ERROR: hard reconstruction MSE did not decrease during GSVQ optimization",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
