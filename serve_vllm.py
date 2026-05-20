#!/usr/bin/env python3
"""
Launch vLLM server after waiting for the Ray cluster to be ready.
Used by scripts/serve_model.sbatch.sh on the head node: waits for all Ray
nodes, starts vllm serve in the background, polls /health, runs one
completions test, then blocks until the server exits.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error


def wait_for_ray(num_nodes, timeout_sec=300, poll_interval=5):
    try:
        import ray
    except ImportError:
        print("ray not installed", file=sys.stderr)
        return False
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            ray.init(address="auto", ignore_reinit_error=True)
            alive = sum(1 for nd in ray.nodes() if nd.get("Alive"))
            ray.shutdown()
            if alive >= num_nodes:
                print(f"Ray cluster ready: {alive} nodes alive.")
                return True
            print(f"  Ray nodes alive: {alive}/{num_nodes} -- waiting...")
        except Exception as e:
            print(f"  Ray wait failed: {e}")
        time.sleep(poll_interval)
    print("Ray cluster did not reach expected size in time.", file=sys.stderr)
    return False


def wait_for_health(port, timeout_sec=900, poll_interval=5):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            r = urllib.request.urlopen(url, timeout=2)
            if r.getcode() == 200:
                print("Server is up.")
                return True
        except Exception:
            pass
        time.sleep(poll_interval)
    print("Health check did not return 200 in time.", file=sys.stderr)
    return False


def run_completions_test(port, max_tokens=32, timeout=30):
    url = f"http://127.0.0.1:{port}/v1/completions"
    data = json.dumps({
        "model": "",
        "prompt": "The capital of France is",
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    r = urllib.request.urlopen(req, timeout=timeout)
    out = r.read().decode()[:600]
    print("Quick completions test:")
    print(out)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-nodes", type=int, required=True, help="Expected number of Ray nodes")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--ray-wait-timeout", type=int, default=300, help="Seconds to wait for Ray cluster")
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=int(os.environ.get("SERVE_HEALTH_MAX_WAIT_SEC", "14400")),
        help="Seconds to wait for /health 200 (default: SERVE_HEALTH_MAX_WAIT_SEC or 14400)",
    )
    args, vllm_args = parser.parse_known_args()
    if not vllm_args:
        parser.error("vllm serve arguments required (e.g. /path/to/model --tensor-parallel-size 4)")

    # Single-node serves don't need (and shouldn't require) a Ray cluster.
    # Ray is only relevant when --num-nodes > 1; otherwise vllm serve runs locally.
    if args.num_nodes > 1:
        if not wait_for_ray(args.num_nodes, timeout_sec=args.ray_wait_timeout):
            sys.exit(1)
    else:
        print(f"Single-node mode (--num-nodes={args.num_nodes}); skipping Ray cluster wait.")

    cmd = ["vllm", "serve"] + vllm_args
    proc = subprocess.Popen(cmd)
    try:
        if wait_for_health(args.port, timeout_sec=args.health_timeout):
            run_completions_test(args.port)
    finally:
        proc.wait()
        if proc.returncode != 0:
            sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
