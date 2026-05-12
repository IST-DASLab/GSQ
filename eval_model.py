"""Run lm-evaluation-harness benchmarks against a vLLM server and log results to WandB.

Assumes a vLLM-compatible server is already running (see scripts/serve_model.sbatch.sh
or start one locally with `vllm serve`).

Usage:
    # Resolve model from training config (latest run)
    python eval_model.py --config configs/local/config.yaml --base-url http://host:8000/v1/completions

    # Specify run ID explicitly
    python eval_model.py --config configs/local/config.yaml --run-id 20260306-143025_a1b2c3 \
        --base-url http://host:8000/v1/completions

    # Point directly at an assembled model directory
    python eval_model.py --model-path /path/to/assembled --base-url http://host:8000/v1/completions

    # Custom tasks
    python eval_model.py --config configs/local/config.yaml --base-url http://host:8000/v1/completions \
        --tasks gsm8k,arc_challenge
"""
import os
import sys
import json
import glob
import argparse
import subprocess

from src.config import load_config, EvalConfig


DEFAULT_TASKS = EvalConfig().default_tasks


def find_latest_run(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for name in os.listdir(checkpoint_dir):
        run_path = os.path.join(checkpoint_dir, name)
        prog_path = os.path.join(run_path, "progress.json")
        if os.path.isdir(run_path) and os.path.exists(prog_path):
            mtime = os.path.getmtime(prog_path)
            candidates.append((mtime, name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def resolve_model_path_from_config(config_path, run_id=None):
    """Resolve the assembled model directory from a training config + optional run ID."""
    cfg = load_config(config_path)
    checkpoint_dir = cfg.training.checkpoint_dir

    if run_id is None:
        run_id = find_latest_run(checkpoint_dir)
        if run_id is None:
            raise RuntimeError(
                f"No completed runs found in '{checkpoint_dir}'. "
                "Pass --run-id explicitly or use --model-path.")

    assembled = os.path.join(checkpoint_dir, run_id, "assembled")
    if not os.path.isdir(assembled):
        raise FileNotFoundError(
            f"Assembled model not found at {assembled}. "
            "Run save_model.py first.")

    return assembled, run_id, checkpoint_dir


def load_wandb_run_id(checkpoint_dir, run_id):
    """Read the WandB run ID from progress.json."""
    prog_path = os.path.join(checkpoint_dir, run_id, "progress.json")
    if not os.path.exists(prog_path):
        return None
    with open(prog_path, "r") as f:
        progress = json.load(f)
    return progress.get("wandb_run_id")


def find_results_file(output_dir):
    """Find the lm-eval results JSON file in the output directory."""
    pattern = os.path.join(output_dir, "**", "results.json")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return max(matches, key=os.path.getmtime)
    json_files = glob.glob(os.path.join(output_dir, "**", "*.json"), recursive=True)
    if json_files:
        return max(json_files, key=os.path.getmtime)
    return None


def parse_results(results_path):
    """Parse lm-eval results JSON and return a dict of task -> metrics."""
    with open(results_path, "r") as f:
        data = json.load(f)

    results = data.get("results", {})
    parsed = {}
    for task_name, metrics in results.items():
        parsed[task_name] = {}
        for key, value in metrics.items():
            if key.startswith("alias"):
                continue
            if isinstance(value, (int, float)):
                parsed[task_name][key] = value
    return parsed


def log_to_wandb(parsed_results, wandb_run_id, config_path):
    """Resume the training WandB run and log evaluation results."""
    from dotenv import load_dotenv
    load_dotenv()
    import wandb

    cfg = load_config(config_path) if config_path else None
    project = cfg.wandb.project if cfg else "gsq"
    wandb_kwargs = dict(project=project)
    if cfg and cfg.wandb.entity:
        wandb_kwargs["entity"] = cfg.wandb.entity
    if wandb_run_id:
        wandb_kwargs["id"] = wandb_run_id
        wandb_kwargs["resume"] = "must"

    try:
        wandb.init(**wandb_kwargs)
    except wandb.errors.CommError as e:
        if wandb_run_id:
            print(
                f"WARNING: Could not resume WandB run '{wandb_run_id}': {e}\n"
                "Falling back to a new WandB run.",
                file=sys.stderr,
            )
            wandb_kwargs.pop("id")
            wandb_kwargs.pop("resume")
            wandb.init(**wandb_kwargs)
        else:
            raise

    flat_metrics = {}
    table_data = []
    for task_name, metrics in parsed_results.items():
        for metric_key, value in metrics.items():
            if ",stderr" in metric_key:
                flat_key = f"eval/{task_name}/{metric_key.replace(',stderr', '_stderr')}"
            else:
                flat_key = f"eval/{task_name}/{metric_key}"
            flat_metrics[flat_key] = value

        primary = metrics.get("acc_norm,none", metrics.get("acc,none",
                  metrics.get("exact_match,strict-match", None)))
        table_data.append([task_name, primary])

    wandb.log(flat_metrics)

    table = wandb.Table(columns=["task", "score"], data=table_data)
    wandb.log({"eval/summary": table})

    wandb.finish()
    return flat_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run lm-eval benchmarks against a vLLM server and log to WandB")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to the assembled model directory (overrides --config resolution)")
    parser.add_argument("--config", type=str, default=None,
                        help="Training config YAML (used to resolve model path and WandB run ID)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Training run ID (defaults to latest)")
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS,
                        help=f"Comma-separated lm-eval tasks (default: {DEFAULT_TASKS})")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1/completions",
                        help="vLLM server completions endpoint")
    parser.add_argument("--num-concurrent", type=int, default=8,
                        help="Number of concurrent requests to vLLM server")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for lm-eval results (default: <model_path>/eval_results)")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="WandB run ID to resume (overrides progress.json lookup)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip WandB logging")
    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
        run_id = args.run_id
        checkpoint_dir = None
    elif args.config:
        model_path, run_id, checkpoint_dir = resolve_model_path_from_config(
            args.config, args.run_id)
    else:
        parser.error("Either --model-path or --config is required")

    output_dir = args.output_dir or os.path.join(model_path, "evals")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer_path = model_path

    print("=" * 50)
    print("GSQ Benchmark Evaluation")
    print(f"  Model path : {model_path}")
    print(f"  Server URL : {args.base_url}")
    print(f"  Tasks      : {args.tasks}")
    print(f"  Concurrent : {args.num_concurrent}")
    print(f"  Output dir : {output_dir}")
    if run_id:
        print(f"  Run ID     : {run_id}")
    print("=" * 50)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "local-completions",
        "--tasks", args.tasks,
        "--model_args", (
            f"model={model_path},"
            f"base_url={args.base_url},"
            f"num_concurrent={args.num_concurrent},"
            f"tokenizer={tokenizer_path}"
        ),
        "--gen_kwargs", "temperature=0,seed=42",
        "--output_path", output_dir,
        "--log_samples",
        "--trust_remote_code",
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nlm-eval exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    results_file = find_results_file(output_dir)
    if results_file is None:
        print("Warning: could not find results JSON in output directory", file=sys.stderr)
        sys.exit(1)

    print(f"\nParsing results from: {results_file}")
    parsed = parse_results(results_file)

    print("\nResults:")
    for task_name, metrics in parsed.items():
        primary = metrics.get("acc_norm,none", metrics.get("acc,none",
                  metrics.get("exact_match,strict-match", "N/A")))
        print(f"  {task_name}: {primary}")

    if not args.no_wandb:
        wandb_run_id = args.wandb_run_id
        if not wandb_run_id and checkpoint_dir and run_id:
            wandb_run_id = load_wandb_run_id(checkpoint_dir, run_id)

        if wandb_run_id:
            print(f"\nLogging to WandB run: {wandb_run_id}")
        else:
            print("\nLogging to new WandB run (no training run ID found)")

        logged = log_to_wandb(parsed, wandb_run_id, args.config)
        print(f"  Logged {len(logged)} metrics to WandB")
    else:
        print("\nSkipping WandB logging (--no-wandb)")

    print("\nDone.")


if __name__ == "__main__":
    main()
