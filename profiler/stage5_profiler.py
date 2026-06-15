"""
TinyMed — Stage 5: Hardware-Aware Profiling Dashboard
Uses torch.profiler to measure:
  - Operator-level latency (CPU time per op)
  - Memory bandwidth utilization
  - FLOP counts (via fvcore or manual)
  - Parameter counts
  - Before/after compression comparison

Outputs an HTML report and a JSON summary for the React dashboard.
"""

import datetime
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

IMG_SIZE = 224
DUMMY = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_models() -> Dict[str, nn.Module]:
    """Load all models for comparison. Falls back gracefully if files missing."""
    model_map = {}

    # Baseline ResNet-18
    baseline = models.resnet18(weights=None)
    baseline.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))
    ckpt_path = MODELS_DIR / "baseline_checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        baseline.load_state_dict(ckpt["model_state_dict"])
    baseline.eval()
    model_map["ResNet-18 Baseline"] = baseline

    # Pruned model
    pruned_path = MODELS_DIR / "model_pruned.pt"
    if pruned_path.exists():
        pruned = models.resnet18(weights=None)
        pruned.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))
        pruned.load_state_dict(torch.load(pruned_path, map_location="cpu"))
        pruned.eval()
        model_map["ResNet-18 Pruned (30%)"] = pruned

    # EfficientNet-B0 student
    student_path = MODELS_DIR / "model_student.pt"
    if student_path.exists():
        student = models.efficientnet_b0(weights=None)
        in_features = student.classifier[1].in_features
        student.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 2))
        student.load_state_dict(torch.load(student_path, map_location="cpu"))
        student.eval()
        model_map["EfficientNet-B0 Student"] = student

    if not model_map:
        log.warning("No trained models found — using fresh ResNet-18 for demo profiling.")
        demo = models.resnet18(weights=None)
        demo.fc = nn.Linear(512, 2)
        demo.eval()
        model_map["ResNet-18 Demo (untrained)"] = demo

    return model_map


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER COUNT
# ═══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nonzero = sum(p.nonzero().size(0) for p in model.parameters())
    return {
        "total_params": total,
        "trainable_params": trainable,
        "nonzero_params": nonzero,
        "sparsity_pct": round((1 - nonzero / total) * 100, 2) if total > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SIZE
# ═══════════════════════════════════════════════════════════════════════════════

def get_model_size_mb(model: nn.Module) -> float:
    tmp = LOGS_DIR / "_tmp_profile_size.pt"
    torch.save(model.state_dict(), tmp)
    size = tmp.stat().st_size / (1024 ** 2)
    tmp.unlink()
    return round(size, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# LATENCY BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def measure_latency(model: nn.Module, n_warmup: int = 20, n_runs: int = 200) -> Dict:
    model.eval()
    dummy = DUMMY.clone()

    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": round(sum(times) / len(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "p50_ms": round(sorted(times)[len(times) // 2], 2),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 2),
        "p99_ms": round(sorted(times)[int(len(times) * 0.99)], 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOP ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_flops(model: nn.Module) -> Dict:
    """
    Estimate FLOPs using fvcore if available, otherwise fall back to a
    parameter-based heuristic. Returns both the value and the source so
    the report can flag heuristic estimates visually.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, DUMMY)
        flops.unsupported_ops_warnings(False)
        return {"value": int(flops.total()), "source": "fvcore"}
    except ImportError:
        params = sum(p.numel() for p in model.parameters())
        estimated = params * 2  # Very rough lower bound — 2 MACs per param
        log.warning(
            "fvcore not installed — FLOPs are a rough heuristic (params x 2). "
            "pip install fvcore for accurate counts."
        )
        return {"value": estimated, "source": "heuristic"}


# ═══════════════════════════════════════════════════════════════════════════════
# TORCH.PROFILER — OPERATOR-LEVEL BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

def profile_model(model: nn.Module, model_name: str) -> List[Dict]:
    """Run torch.profiler and extract top-20 CPU ops by self CPU time."""
    model.eval()
    dummy = DUMMY.clone()

    trace_dir = LOGS_DIR / "profiler_traces" / model_name.replace(" ", "_").replace("/", "_")
    trace_dir.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                for _ in range(20):   # Profile 20 runs for stability
                    _ = model(dummy)

    # Extract top ops — exclude record_function annotation spans
    EXCLUDE_OPS = {"model_inference"}
    events = prof.key_averages()
    top_ops = []
    for evt in sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True):
        if evt.key in EXCLUDE_OPS:
            continue
        top_ops.append({
            "op": evt.key,
            "self_cpu_ms": round(evt.self_cpu_time_total / 1e3 / 20, 4),  # Per inference
            "cpu_ms": round(evt.cpu_time_total / 1e3 / 20, 4),
            "calls": evt.count // 20,
            "self_memory_mb": round(evt.self_cpu_memory_usage / (1024 ** 2), 4),
            "flops": int(evt.flops) if hasattr(evt, "flops") else 0,
        })
        if len(top_ops) == 20:
            break

    return top_ops


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY BANDWIDTH ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_memory_bandwidth(model: nn.Module, latency_ms: float) -> Dict:
    """
    Estimate memory bandwidth = (model_params * bytes_per_param) / latency.
    This is the 'roofline model' floor — actual bandwidth depends on hardware.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # Input + output tensors
    input_bytes = DUMMY.numel() * DUMMY.element_size()
    total_bytes = param_bytes + input_bytes * 2

    bandwidth_gb_s = (total_bytes / (1024 ** 3)) / (latency_ms / 1000)
    return {
        "param_memory_mb": round(param_bytes / (1024 ** 2), 2),
        "total_io_memory_mb": round(total_bytes / (1024 ** 2), 2),
        "estimated_bandwidth_gb_s": round(bandwidth_gb_s, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html_report(all_results: List[Dict], run_meta: Dict) -> Path:
    # ── Fix 4: timestamped filename so runs never overwrite each other ────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = LOGS_DIR / f"profiling_report_{ts}.html"

    # ── Fix 5 & 6: resolve latency SLA from env or default ───────────────────
    # Set TINYMED_LATENCY_SLA_MS in your environment to pin absolute thresholds.
    import os
    SLA_MS = float(os.environ.get("TINYMED_LATENCY_SLA_MS", 0))  # 0 = use relative

    def latency_class(mean_ms: float, pn_ms: float) -> str:
        """
        If SLA_MS is set: absolute threshold (red if > SLA, yellow if > 0.85*SLA).
        Otherwise: relative to mean (>2x red, >1.4x yellow).
        """
        if SLA_MS > 0:
            if pn_ms > SLA_MS:          return "latency-crit"
            if pn_ms > SLA_MS * 0.85:   return "latency-warn"
            return ""
        ratio = pn_ms / mean_ms if mean_ms > 0 else 0
        if ratio > 2.0:  return "latency-crit"
        if ratio > 1.4:  return "latency-warn"
        return ""

    # ── Build model summary rows ──────────────────────────────────────────────
    rows = ""
    for r in all_results:
        lat  = r["latency"]["mean_ms"]
        p95  = r["latency"]["p95_ms"]
        p99  = r["latency"]["p99_ms"]
        cls95 = latency_class(lat, p95)
        cls99 = latency_class(lat, p99)
        p95_td = f'<td class="{cls95}">{p95} ms</td>' if cls95 else f"<td>{p95} ms</td>"
        p99_td = f'<td class="{cls99}">{p99} ms</td>' if cls99 else f"<td>{p99} ms</td>"

        # Fix 1: label FLOPs with source badge
        flop_val = r["flops"] / 1e6
        flop_src = r.get("flops_source", "unknown")
        if flop_src == "heuristic":
            flop_td = (f'<td><span class="flop-heuristic" ' +
                       f'title="Rough estimate: params x 2. Install fvcore for accuracy.">' +
                       f'{flop_val:.1f}M &#9888;</span></td>')
        else:
            flop_td = f"<td>{flop_val:.1f}M</td>"

        rows += f"""
        <tr>
            <td><strong>{r["model_name"]}</strong></td>
            <td>{r["size_mb"]} MB</td>
            <td>{r["params"]["total_params"]:,}</td>
            <td>{r["params"]["sparsity_pct"]}%</td>
            <td>{lat} ms</td>
            {p95_td}
            {p99_td}
            {flop_td}
            <td>{r["memory"]["estimated_bandwidth_gb_s"]} GB/s</td>
        </tr>"""

    # ── Build op breakdown tables ─────────────────────────────────────────────
    op_tables = ""
    num_models = len(all_results)
    for r in all_results:
        model_name = r["model_name"]
        op_rows = ""

        # Fix 4: slow-conv banner — also checks mkldnn/mps availability
        slow_ops = [op for op in r.get("top_ops", []) if "_slow_conv2d" in op["op"]]
        if slow_ops:
            slow_ms = slow_ops[0]["self_cpu_ms"]
            mean_ms = r["latency"]["mean_ms"]
            pct     = round(slow_ms / mean_ms * 100, 1) if mean_ms else 0
            if run_meta.get("mps_available"):
                fix_advice = (
                    "This machine has MPS (Apple Silicon). "
                    "Move the model and inputs to MPS: "
                    "<code>model.to('mps')</code> and <code>dummy.to('mps')</code>."
                )
            elif run_meta.get("mkldnn_enabled"):
                fix_advice = (
                    "MKL-DNN is available but not being used. Ensure inputs are "
                    "contiguous float32: <code>x = x.contiguous()</code>. "
                    "Also try <code>torch.compile(model)</code>."
                )
            else:
                fix_advice = (
                    "Neither MKL-DNN nor MPS is available on this machine. "
                    "Consider moving inference to a machine with AVX2/MKL support "
                    "or a CUDA GPU."
                )
            warn_banner = f"""
        <div class="perf-warning">
          <span style="color:var(--red);font-size:16px;">&#9888;</span>
          <div>
            <strong>Performance Warning &mdash; Slow Conv Path Detected</strong>
            <p>
              <code>aten::_slow_conv2d_forward</code> accounts for
              <strong>{slow_ms} ms</strong> self-CPU time ({pct}% of mean inference).
              PyTorch is falling back to a generic CPU kernel instead of an
              optimised backend (MKL-DNN / cuDNN / MPS).<br><br>
              <strong>Fix for this machine:</strong> {fix_advice}
            </p>
          </div>
        </div>"""
        else:
            warn_banner = ""

        for op in r.get("top_ops", []):
            # Fix 1: dim zero-FLOP cells
            flops_val = op["flops"]
            if flops_val == 0:
                flops_td = '<td class="flop-zero">&mdash;</td>'
            else:
                flops_td = f"<td>{flops_val:,}</td>"

            # Fix 2: tooltip only for meaningfully negative memory (< -0.01 MB)
            mem_val = op["self_memory_mb"]
            if mem_val < -0.01:
                mem_td = (
                    f'<td><span class="mem-negative" ' +
                    f'title="Negative = memory freed/returned to allocator. ' +
                    f'Expected behaviour in PyTorch memory accounting.">' +
                    f'{mem_val:.4f}</span></td>'
                )
            else:
                mem_td = f"<td>{mem_val:.4f}</td>"

            op_rows += f"""
            <tr>
                <td class="op-name">{op["op"]}</td>
                <td>{op["self_cpu_ms"]:.4f}</td>
                <td>{op["cpu_ms"]:.4f}</td>
                <td>{op["calls"]}</td>
                {flops_td}
                {mem_td}
            </tr>"""

        op_tables += f"""
        {warn_banner}
        <div class="op-section">
            <h3>{model_name} — Top Operators</h3>
            <table class="op-table">
                <thead>
                    <tr>
                        <th>Operator</th>
                        <th>Self CPU (ms)</th>
                        <th>Total CPU (ms)</th>
                        <th>Calls</th>
                        <th>FLOPs <span style="color:var(--accent2)">*</span></th>
                        <th>Self Mem (MB)</th>
                    </tr>
                </thead>
                <tbody>{op_rows}</tbody>
            </table>
            <p class="flop-note">
              * FLOPs attributed only at top-level dispatch ops (conv2d, addmm).
              Child ops show &mdash; by design &mdash; not zero compute.
            </p>
        </div>"""

    # ── Fix 3: heading pluralises only when multiple models present ───────────
    breakdown_heading = (
        f"Operator-Level Breakdown &mdash; Top 20 Ops per Model"
        if num_models > 1 else
        "Operator-Level Breakdown &mdash; Top 20 Ops"
    )

    # ── Fix 4: run metadata block ─────────────────────────────────────────────
    sla_note = (f"SLA: {SLA_MS} ms (absolute)" if SLA_MS > 0
                else "Latency thresholds: relative to mean (warn &gt;1.4&times;, crit &gt;2.0&times;)")
    mkldnn_badge = ('<span class="badge-ok">MKL-DNN available</span>'
                    if run_meta.get("mkldnn_enabled")
                    else '<span class="badge-warn">MKL-DNN unavailable</span>')
    mps_badge = ('<span class="badge-ok">MPS available</span>'
                 if run_meta.get("mps_available")
                 else '<span class="badge-muted">MPS unavailable</span>')

    meta_block = f"""
    <section>
      <h2>Run Metadata</h2>
      <table>
        <tbody>
          <tr><td>Generated</td><td>{run_meta["timestamp"]}</td></tr>
          <tr><td>PyTorch</td><td>{run_meta["pytorch_version"]}</td></tr>
          <tr><td>Python</td><td>{run_meta["python_version"]}</td></tr>
          <tr><td>Platform</td><td>{run_meta["platform"]}</td></tr>
          <tr><td>Backends</td><td>{mkldnn_badge} {mps_badge}</td></tr>
          <tr><td>Latency thresholds</td><td>{sla_note}</td></tr>
        </tbody>
      </table>
    </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TinyMed — Profiling Report</title>
<style>
  :root {{
    --bg: #0a0e1a;
    --surface: #111827;
    --border: #1f2937;
    --accent: #00d9ff;
    --accent2: #7c3aed;
    --text: #e5e7eb;
    --muted: #6b7280;
    --green: #10b981;
    --yellow: #f59e0b;
    --red: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 13px;
    line-height: 1.6;
  }}
  header {{
    border-bottom: 1px solid var(--border);
    padding: 32px 48px;
    display: flex;
    align-items: baseline;
    gap: 24px;
  }}
  header h1 {{
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.5px;
  }}
  header span {{
    color: var(--muted);
    font-size: 12px;
  }}
  main {{ padding: 48px; max-width: 1400px; margin: 0 auto; }}
  section {{ margin-bottom: 56px; }}
  h2 {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
  }}
  thead th {{
    text-align: left;
    padding: 12px 16px;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    background: #0d1424;
  }}
  tbody td {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover td {{ background: rgba(0, 217, 255, 0.03); }}
  .op-name {{ color: var(--accent); font-size: 11px; max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .op-section {{ margin-bottom: 40px; }}
  .op-section h3 {{ font-size: 13px; color: var(--text); margin-bottom: 12px; }}
  .op-table thead th {{ font-size: 9px; }}
  .highlight {{ color: var(--accent); font-weight: 600; }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    background: rgba(0, 217, 255, 0.1);
    color: var(--accent);
    margin-left: 8px;
  }}
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }}
  .metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
  }}
  .metric-card .label {{
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  .metric-card .value {{
    font-size: 24px;
    font-weight: 700;
    color: var(--accent);
  }}
  .metric-card .sub {{
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
  }}
  .latency-warn {{ color: var(--yellow); font-weight: 600; }}
  .latency-crit {{ color: var(--red);    font-weight: 600; }}
  .mem-negative {{ color: var(--muted); font-style: italic; cursor: help; border-bottom: 1px dashed var(--muted); }}
  .flop-zero {{ color: var(--muted); }}
  .flop-heuristic {{ color: var(--yellow); cursor: help; border-bottom: 1px dashed var(--yellow); }}
  .flop-note {{ font-size: 11px; color: var(--muted); margin-top: 8px; }}
  .perf-warning {{ background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3); border-radius: 8px; padding: 16px 20px; margin-bottom: 24px; display: flex; gap: 12px; align-items: flex-start; }}
  .perf-warning strong {{ color: var(--red); }}
  .perf-warning p {{ color: var(--muted); margin-top: 4px; font-size: 12px; }}
  .badge-ok   {{ background: rgba(16,185,129,0.12); color: var(--green);  padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
  .badge-warn {{ background: rgba(245,158,11,0.12);  color: var(--yellow); padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
  .badge-muted {{ background: rgba(107,114,128,0.12); color: var(--muted); padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
</style>
</head>
<body>
<header>
  <h1>TinyMed Profiling Report</h1>
  <span>Hardware-Aware Model Analysis · Generated by torch.profiler</span>
</header>
<main>

{meta_block}

<section>
  <h2>Model Profile Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Size</th>
        <th>Params</th>
        <th>Sparsity</th>
        <th>Mean Latency</th>
        <th>P95 Latency</th>
        <th>P99 Latency</th>
        <th>FLOPs</th>
        <th>Est. BW</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</section>

<section>
  <h2>{breakdown_heading}</h2>
  {op_tables}
</section>

</main>
</body>
</html>"""

    html_path.write_text(html)
    log.info(f"HTML report saved to {html_path}")
    return html_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    models_to_profile = load_models()
    all_results = []

    for model_name, model in models_to_profile.items():
        log.info(f"\nProfiling: {model_name}")
        log.info("-" * 50)

        params = count_parameters(model)
        size_mb = get_model_size_mb(model)
        latency = measure_latency(model)
        flops_info = estimate_flops(model)
        flops = flops_info["value"]
        flops_source = flops_info["source"]
        memory = estimate_memory_bandwidth(model, latency["mean_ms"])

        log.info(f"  Params: {params['total_params']:,} | Size: {size_mb} MB | "
                 f"Latency: {latency['mean_ms']} ms | "
                 f"FLOPs: {flops/1e6:.1f}M ({flops_source})")

        log.info(f"  Running torch.profiler...")
        try:
            top_ops = profile_model(model, model_name)
        except Exception as e:
            log.warning(f"  Profiler failed: {e}")
            top_ops = []

        result = {
            "model_name": model_name,
            "size_mb": size_mb,
            "params": params,
            "latency": latency,
            "flops": flops,
            "flops_source": flops_source,
            "memory": memory,
            "top_ops": top_ops,
        }
        all_results.append(result)

    # Save JSON
    json_path = LOGS_DIR / "profiling_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nProfiling JSON saved to {json_path}")

    # Generate HTML report
    run_meta = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "mkldnn_enabled": torch.backends.mkldnn.is_available(),
        "mps_available": getattr(torch.backends, 'mps', None) is not None
                          and torch.backends.mps.is_available(),
    }
    html_path = generate_html_report(all_results, run_meta)

    # Print terminal summary
    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'Size':>8} {'Latency':>10} {'FLOPs':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model_name']:<35} {r['size_mb']:>7.2f}MB "
              f"{r['latency']['mean_ms']:>9.2f}ms "
              f"{r['flops']/1e6:>9.1f}M")
    print("=" * 70)
    print(f"\nFull report: {html_path}")
    print(f"JSON data:   {json_path}")
    print(f"Traces:      {LOGS_DIR / 'profiler_traces'} (open in TensorBoard)")
    print(f"  → tensorboard --logdir {LOGS_DIR / 'profiler_traces'}")


if __name__ == "__main__":
    main()

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_models() -> Dict[str, nn.Module]:
    """Load all models for comparison. Falls back gracefully if files missing."""
    model_map = {}

    # Baseline ResNet-18
    baseline = models.resnet18(weights=None)
    baseline.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))
    ckpt_path = MODELS_DIR / "baseline_checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        baseline.load_state_dict(ckpt["model_state_dict"])
    baseline.eval()
    model_map["ResNet-18 Baseline"] = baseline

    # Pruned model
    pruned_path = MODELS_DIR / "model_pruned.pt"
    if pruned_path.exists():
        pruned = models.resnet18(weights=None)
        pruned.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))
        pruned.load_state_dict(torch.load(pruned_path, map_location="cpu"))
        pruned.eval()
        model_map["ResNet-18 Pruned (30%)"] = pruned

    # EfficientNet-B0 student
    student_path = MODELS_DIR / "model_student.pt"
    if student_path.exists():
        student = models.efficientnet_b0(weights=None)
        in_features = student.classifier[1].in_features
        student.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 2))
        student.load_state_dict(torch.load(student_path, map_location="cpu"))
        student.eval()
        model_map["EfficientNet-B0 Student"] = student

    if not model_map:
        log.warning("No trained models found — using fresh ResNet-18 for demo profiling.")
        demo = models.resnet18(weights=None)
        demo.fc = nn.Linear(512, 2)
        demo.eval()
        model_map["ResNet-18 Demo (untrained)"] = demo

    return model_map


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER COUNT
# ═══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nonzero = sum(p.nonzero().size(0) for p in model.parameters())
    return {
        "total_params": total,
        "trainable_params": trainable,
        "nonzero_params": nonzero,
        "sparsity_pct": round((1 - nonzero / total) * 100, 2) if total > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SIZE
# ═══════════════════════════════════════════════════════════════════════════════

def get_model_size_mb(model: nn.Module) -> float:
    tmp = LOGS_DIR / "_tmp_profile_size.pt"
    torch.save(model.state_dict(), tmp)
    size = tmp.stat().st_size / (1024 ** 2)
    tmp.unlink()
    return round(size, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# LATENCY BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def measure_latency(model: nn.Module, n_warmup: int = 20, n_runs: int = 200) -> Dict:
    model.eval()
    dummy = DUMMY.clone()

    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": round(sum(times) / len(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "p50_ms": round(sorted(times)[len(times) // 2], 2),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 2),
        "p99_ms": round(sorted(times)[int(len(times) * 0.99)], 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOP ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_flops(model: nn.Module) -> int:
    """
    Estimate FLOPs using fvcore if available, otherwise use a heuristic
    based on parameter count and input size.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, DUMMY)
        flops.unsupported_ops_warnings(False)
        return int(flops.total())
    except ImportError:
        # Heuristic: for a typical CNN, FLOPs ≈ 2 × params × (spatial_ops_factor)
        params = sum(p.numel() for p in model.parameters())
        estimated = params * 2  # Very rough lower bound
        log.debug("fvcore not installed — using parameter-based FLOP estimate. "
                  "pip install fvcore for accurate counts.")
        return estimated


# ═══════════════════════════════════════════════════════════════════════════════
# TORCH.PROFILER — OPERATOR-LEVEL BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

def profile_model(model: nn.Module, model_name: str) -> List[Dict]:
    """Run torch.profiler and extract top-20 CPU ops by self CPU time."""
    model.eval()
    dummy = DUMMY.clone()

    trace_dir = LOGS_DIR / "profiler_traces" / model_name.replace(" ", "_").replace("/", "_")
    trace_dir.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                for _ in range(20):   # Profile 20 runs for stability
                    _ = model(dummy)

    # Extract top ops
    events = prof.key_averages()
    top_ops = []
    for evt in sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:20]:
        top_ops.append({
            "op": evt.key,
            "self_cpu_ms": round(evt.self_cpu_time_total / 1e3 / 20, 4),  # Per inference
            "cpu_ms": round(evt.cpu_time_total / 1e3 / 20, 4),
            "calls": evt.count // 20,
            "self_memory_mb": round(evt.self_cpu_memory_usage / (1024 ** 2), 4),
            "flops": int(evt.flops) if hasattr(evt, "flops") else 0,
        })

    return top_ops


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY BANDWIDTH ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_memory_bandwidth(model: nn.Module, latency_ms: float) -> Dict:
    """
    Estimate memory bandwidth = (model_params * bytes_per_param) / latency.
    This is the 'roofline model' floor — actual bandwidth depends on hardware.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # Input + output tensors
    input_bytes = DUMMY.numel() * DUMMY.element_size()
    total_bytes = param_bytes + input_bytes * 2

    bandwidth_gb_s = (total_bytes / (1024 ** 3)) / (latency_ms / 1000)
    return {
        "param_memory_mb": round(param_bytes / (1024 ** 2), 2),
        "total_io_memory_mb": round(total_bytes / (1024 ** 2), 2),
        "estimated_bandwidth_gb_s": round(bandwidth_gb_s, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html_report(all_results: List[Dict]) -> Path:
    html_path = LOGS_DIR / "profiling_report.html"

    # Build model summary rows
    rows = ""
    for r in all_results:
        lat = r["latency"]["mean_ms"]
        rows += f"""
        <tr>
            <td><strong>{r["model_name"]}</strong></td>
            <td>{r["size_mb"]} MB</td>
            <td>{r["params"]["total_params"]:,}</td>
            <td>{r["params"]["sparsity_pct"]}%</td>
            <td>{lat} ms</td>
            <td>{r["latency"]["p95_ms"]} ms</td>
            <td>{r["latency"]["p99_ms"]} ms</td>
            <td>{r["flops"] / 1e6:.1f}M</td>
            <td>{r["memory"]["estimated_bandwidth_gb_s"]} GB/s</td>
        </tr>"""

    # Build op breakdown tables
    op_tables = ""
    for r in all_results:
        model_name = r["model_name"]
        op_rows = ""
        for op in r.get("top_ops", []):
            op_rows += f"""
            <tr>
                <td class="op-name">{op["op"]}</td>
                <td>{op["self_cpu_ms"]:.4f}</td>
                <td>{op["cpu_ms"]:.4f}</td>
                <td>{op["calls"]}</td>
                <td>{op["flops"]:,}</td>
                <td>{op["self_memory_mb"]:.4f}</td>
            </tr>"""

        op_tables += f"""
        <div class="op-section">
            <h3>{model_name} — Top Operators</h3>
            <table class="op-table">
                <thead>
                    <tr>
                        <th>Operator</th>
                        <th>Self CPU (ms)</th>
                        <th>Total CPU (ms)</th>
                        <th>Calls</th>
                        <th>FLOPs</th>
                        <th>Self Mem (MB)</th>
                    </tr>
                </thead>
                <tbody>{op_rows}</tbody>
            </table>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TinyMed — Profiling Report</title>
<style>
  :root {{
    --bg: #0a0e1a;
    --surface: #111827;
    --border: #1f2937;
    --accent: #00d9ff;
    --accent2: #7c3aed;
    --text: #e5e7eb;
    --muted: #6b7280;
    --green: #10b981;
    --yellow: #f59e0b;
    --red: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 13px;
    line-height: 1.6;
  }}
  header {{
    border-bottom: 1px solid var(--border);
    padding: 32px 48px;
    display: flex;
    align-items: baseline;
    gap: 24px;
  }}
  header h1 {{
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.5px;
  }}
  header span {{
    color: var(--muted);
    font-size: 12px;
  }}
  main {{ padding: 48px; max-width: 1400px; margin: 0 auto; }}
  section {{ margin-bottom: 56px; }}
  h2 {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
  }}
  thead th {{
    text-align: left;
    padding: 12px 16px;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    background: #0d1424;
  }}
  tbody td {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover td {{ background: rgba(0, 217, 255, 0.03); }}
  .op-name {{ color: var(--accent); font-size: 11px; max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .op-section {{ margin-bottom: 40px; }}
  .op-section h3 {{ font-size: 13px; color: var(--text); margin-bottom: 12px; }}
  .op-table thead th {{ font-size: 9px; }}
  .highlight {{ color: var(--accent); font-weight: 600; }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    background: rgba(0, 217, 255, 0.1);
    color: var(--accent);
    margin-left: 8px;
  }}
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }}
  .metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
  }}
  .metric-card .label {{
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  .metric-card .value {{
    font-size: 24px;
    font-weight: 700;
    color: var(--accent);
  }}
  .metric-card .sub {{
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
  }}
</style>
</head>
<body>
<header>
  <h1>TinyMed Profiling Report</h1>
  <span>Hardware-Aware Model Analysis · Generated by torch.profiler</span>
</header>
<main>

<section>
  <h2>Model Comparison Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Size</th>
        <th>Params</th>
        <th>Sparsity</th>
        <th>Mean Latency</th>
        <th>P95 Latency</th>
        <th>P99 Latency</th>
        <th>FLOPs</th>
        <th>Est. BW</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</section>

<section>
  <h2>Operator-Level Breakdown (Top 20 ops per model)</h2>
  {op_tables}
</section>

</main>
</body>
</html>"""

    html_path.write_text(html)
    log.info(f"HTML report saved to {html_path}")
    return html_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    models_to_profile = load_models()
    all_results = []

    for model_name, model in models_to_profile.items():
        log.info(f"\nProfiling: {model_name}")
        log.info("-" * 50)

        params = count_parameters(model)
        size_mb = get_model_size_mb(model)
        latency = measure_latency(model)
        flops = estimate_flops(model)
        memory = estimate_memory_bandwidth(model, latency["mean_ms"])

        log.info(f"  Params: {params['total_params']:,} | Size: {size_mb} MB | "
                 f"Latency: {latency['mean_ms']} ms | FLOPs: {flops/1e6:.1f}M")

        log.info(f"  Running torch.profiler...")
        try:
            top_ops = profile_model(model, model_name)
        except Exception as e:
            log.warning(f"  Profiler failed: {e}")
            top_ops = []

        result = {
            "model_name": model_name,
            "size_mb": size_mb,
            "params": params,
            "latency": latency,
            "flops": flops,
            "memory": memory,
            "top_ops": top_ops,
        }
        all_results.append(result)

    # Save JSON
    json_path = LOGS_DIR / "profiling_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nProfiling JSON saved to {json_path}")

    # Generate HTML report
    html_path = generate_html_report(all_results)

    # Print terminal summary
    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'Size':>8} {'Latency':>10} {'FLOPs':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model_name']:<35} {r['size_mb']:>7.2f}MB "
              f"{r['latency']['mean_ms']:>9.2f}ms "
              f"{r['flops']/1e6:>9.1f}M")
    print("=" * 70)
    print(f"\nFull report: {html_path}")
    print(f"JSON data:   {json_path}")
    print(f"Traces:      {LOGS_DIR / 'profiler_traces'} (open in TensorBoard)")
    print(f"  → tensorboard --logdir {LOGS_DIR / 'profiler_traces'}")


if __name__ == "__main__":
    main()
