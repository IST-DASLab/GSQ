import os


_ENABLED = os.environ.get("GSQ_PROGRESS", "1") != "0"

_SEP = "─" * 60

_GUMBEL_STEP_LOG_INTERVAL = 50
_PPL_REPORT_DIVISOR = 10


def init(config):
    global _GUMBEL_STEP_LOG_INTERVAL, _PPL_REPORT_DIVISOR
    _GUMBEL_STEP_LOG_INTERVAL = config.logging.gumbel_step_log_interval
    _PPL_REPORT_DIVISOR = config.logging.ppl_report_divisor


def _fmt_time(seconds):
    if seconds is None or seconds < 0:
        return "--:--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _pct(current, total):
    if total <= 0:
        return 0.0
    return 100.0 * current / total


def _eta(elapsed, current, total):
    if current <= 0 or total <= 0:
        return None
    return elapsed * (total - current) / current


def report_layer(layer_idx, num_layers, stage, elapsed_total, **kw):
    if not _ENABLED:
        return
    pct = _pct(layer_idx + 1, num_layers)
    eta = _eta(elapsed_total, layer_idx + 1, num_layers)
    lines = [
        _SEP,
        f"  Layer {layer_idx + 1} / {num_layers}  ({pct:.1f}%)  |  Stage: {stage}",
        f"  Elapsed: {_fmt_time(elapsed_total)}  ETA: {_fmt_time(eta)}",
    ]
    for key, val in kw.items():
        lines.append(f"  {key}: {val}")
    lines.append(_SEP)
    print("\n".join(lines), flush=True)


def report_gptq_calib(j, total, elapsed):
    if not _ENABLED:
        return
    eta = _eta(elapsed, j, total)
    print(f"  [GPTQ] Calibration: {j}/{total}  "
          f"({_pct(j, total):.0f}%)  "
          f"Elapsed: {_fmt_time(elapsed)}  ETA: {_fmt_time(eta)}", flush=True)


def report_gptq_linear(idx, total, name, elapsed):
    if not _ENABLED:
        return
    print(f"  [GPTQ] Linear {idx}/{total}: {name}  "
          f"Elapsed: {_fmt_time(elapsed)}", flush=True)


def report_gptq_done(elapsed, avg_loss=None):
    if not _ENABLED:
        return
    msg = f"  [GPTQ] Done in {_fmt_time(elapsed)}"
    if avg_loss is not None:
        msg += f"  avg_loss={avg_loss:.4f}"
    print(msg, flush=True)


def report_gumbel_epoch(layer_name, epoch, num_epochs, phase_elapsed,
                        avg_train_loss=None, avg_val_loss=None,
                        epoch_time=None, temperature=None, scale=None):
    if not _ENABLED:
        return
    eta = _eta(phase_elapsed, epoch + 1, num_epochs)
    parts = [
        f"  [Gumbel] {layer_name}  Epoch {epoch + 1}/{num_epochs}",
        f"  Phase elapsed: {_fmt_time(phase_elapsed)}  ETA: {_fmt_time(eta)}",
    ]
    metrics = []
    if avg_train_loss is not None:
        metrics.append(f"train_loss={avg_train_loss:.2e}")
    if avg_val_loss is not None:
        metrics.append(f"val_hard_loss={avg_val_loss:.2e}")
    if epoch_time is not None:
        metrics.append(f"epoch_time={epoch_time:.1f}s")
    if temperature is not None:
        metrics.append(f"temp={temperature:.3f}")
    if scale is not None:
        metrics.append(f"scale={scale:.1f}")
    if metrics:
        parts.append("  " + "  ".join(metrics))
    print("\n".join(parts), flush=True)


def report_gumbel_step(step, total_steps, loss, interval=None):
    if not _ENABLED:
        return
    if interval is None:
        interval = _GUMBEL_STEP_LOG_INTERVAL
    if (step + 1) % interval != 0 and step + 1 != total_steps:
        return
    print(f"  [Gumbel] Step {step + 1}/{total_steps}  loss={loss:.2e}", flush=True)


def report_throughput(layer_idx, num_layers, layer_time, gptq_time, train_time,
                      num_experts=None, avg_layer_time=None):
    if not _ENABLED:
        return
    parts = [
        _SEP,
        f"  Throughput — Layer {layer_idx + 1}/{num_layers}",
        f"    sec/layer:  {layer_time:.1f}s  (GPTQ {gptq_time:.1f}s + Gumbel {train_time:.1f}s)",
    ]
    if num_experts is not None and num_experts > 0:
        sec_per_expert = layer_time / num_experts
        parts.append(f"    sec/expert: {sec_per_expert:.2f}s  ({num_experts} experts)")
    if avg_layer_time is not None:
        remaining = num_layers - (layer_idx + 1)
        est_remaining = avg_layer_time * remaining
        parts.append(f"    avg sec/layer: {avg_layer_time:.1f}s  "
                     f"est. remaining: {_fmt_time(est_remaining)}")
    parts.append(_SEP)
    print("\n".join(parts), flush=True)


def report_ppl_layer(i, num_layers, elapsed=None):
    if not _ENABLED:
        return
    if (i + 1) % max(1, num_layers // _PPL_REPORT_DIVISOR) != 0 and i + 1 != num_layers:
        return
    msg = f"  [PPL eval] Layer {i + 1}/{num_layers}"
    if elapsed is not None:
        msg += f"  Elapsed: {_fmt_time(elapsed)}"
    print(msg, flush=True)


def report_pipeline(message):
    if not _ENABLED:
        return
    print(f"  [Pipeline] {message}", flush=True)


def report_expert_gptq(expert_idx, total_experts, expert_id, elapsed=None):
    if not _ENABLED:
        return
    msg = f"  [GPTQ] Expert {expert_idx + 1}/{total_experts} (id={expert_id})"
    if elapsed is not None:
        msg += f"  Elapsed: {_fmt_time(elapsed)}"
    print(msg, flush=True)
