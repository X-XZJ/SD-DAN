"""Computational cost metrics for deployment analysis and paper reporting."""

import datetime
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def resolve_input_shape(cfg, model=None) -> Tuple[int, int, int]:
    """Return (batch_size, height, width) for complexity / inference benchmarks."""
    height, width = 800, 1333
    if model is not None:
        if hasattr(model, "min_size") and model.min_size is not None:
            height = int(model.min_size if not isinstance(model.min_size, (list, tuple)) else model.min_size[0])
        if hasattr(model, "max_size") and model.max_size is not None:
            width = int(model.max_size)
    batch_size = getattr(cfg, "batch_size", 1)
    return batch_size, height, width


def get_gpu_memory_mb() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_MB": 0.0, "reserved_MB": 0.0, "max_allocated_MB": 0.0}
    return {
        "allocated_MB": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved_MB": torch.cuda.memory_reserved() / (1024 ** 2),
        "max_allocated_MB": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def reset_peak_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    height: int = 800,
    width: int = 1333,
    warmup: int = 10,
    num_runs: int = 50,
) -> Dict[str, Any]:
    """
    Measure single-image inference latency, FPS, and GPU memory.

    The detector expects a tuple of image tensors: model((image,)).
    """
    result: Dict[str, Any] = {
        "input_height": height,
        "input_width": width,
        "warmup_iters": warmup,
        "benchmark_iters": num_runs,
        "device": str(device),
        "latency_ms_mean": 0.0,
        "latency_ms_std": 0.0,
        "latency_ms_median": 0.0,
        "latency_ms_min": 0.0,
        "latency_ms_max": 0.0,
        "fps": 0.0,
        "gpu_memory_peak_MB": 0.0,
        "gpu_memory_allocated_MB": 0.0,
        "success": False,
        "error": None,
    }

    was_training = model.training
    model.eval()

    try:
        eval_model = model
        if hasattr(model, "module"):
            eval_model = model.module

        image = torch.randn(3, height, width, device=device)

        with torch.inference_mode():
            _ = eval_model((image,))
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            reset_peak_gpu_memory()

        with torch.inference_mode():
            for _ in range(warmup):
                _ = eval_model((image,))
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        latencies = []
        use_cuda_events = device.type == "cuda"
        starter = ender = None
        if use_cuda_events:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

        with torch.inference_mode():
            for _ in range(num_runs):
                if use_cuda_events:
                    starter.record()
                    _ = eval_model((image,))
                    ender.record()
                    torch.cuda.synchronize(device)
                    latencies.append(starter.elapsed_time(ender))
                else:
                    import time

                    t0 = time.perf_counter()
                    _ = eval_model((image,))
                    latencies.append((time.perf_counter() - t0) * 1000.0)

        if not latencies:
            raise RuntimeError("No latency samples collected")

        latency_tensor = torch.tensor(latencies, dtype=torch.float64)
        mean_ms = latency_tensor.mean().item()
        result.update(
            {
                "latency_ms_mean": mean_ms,
                "latency_ms_std": latency_tensor.std(unbiased=False).item(),
                "latency_ms_median": latency_tensor.median().item(),
                "latency_ms_min": latency_tensor.min().item(),
                "latency_ms_max": latency_tensor.max().item(),
                "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
                "success": True,
            }
        )

        mem = get_gpu_memory_mb()
        result["gpu_memory_peak_MB"] = mem["max_allocated_MB"]
        result["gpu_memory_allocated_MB"] = mem["allocated_MB"]
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        model.train(was_training)

    return result


def summarize_training_epoch(metric_logger) -> Dict[str, float]:
    stats = {}
    if metric_logger is None:
        return stats
    for key in ("iter_time", "data_time"):
        if key in metric_logger.meters:
            stats[f"{key}_avg_s"] = metric_logger.meters[key].global_avg
    if "iter_time" in stats and stats["iter_time_avg_s"] > 0:
        stats["throughput_imgs_per_s"] = 1.0 / stats["iter_time_avg_s"]
    return stats


def format_computational_cost_report(
    params_info: Dict[str, Any],
    flops_info: Dict[str, Any],
    memory_info: Dict[str, Any],
    inference_info: Optional[Dict[str, Any]],
    training_info: Dict[str, Any],
    input_shape: Tuple[int, ...],
) -> str:
    lines = [
        "=" * 80,
        "Computational Cost Report / 计算开销报告",
        "=" * 80,
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input shape (B,C,H,W): {input_shape}",
        "",
        "[1] Parameters / 参数量",
        f"  Total params:      {params_info['total']:,} ({params_info['total_M']:.2f} M)",
        f"  Trainable params:  {params_info['trainable']:,} ({params_info['trainable_M']:.2f} M)",
        f"  Frozen params:     {params_info['non_trainable']:,} ({params_info['non_trainable_M']:.2f} M)",
        "",
        "[2] FLOPs / 计算量",
        f"  Total FLOPs:       {flops_info['flops_G']:.2f} GFLOPs",
        f"  FLOPs/Param ratio: {flops_info['ratio']:.2f} GFLOPs/M",
    ]
    if "method" in flops_info:
        lines.append(f"  Method:            {flops_info['method']}")
    if "note" in flops_info:
        lines.append(f"  Note:              {flops_info['note']}")

    lines.extend(
        [
            "",
            "[3] Memory / 内存消耗",
            f"  Param memory (est):      {memory_info['params_memory_MB']:.2f} MB",
            f"  Activation memory (est): {memory_info['activation_memory_MB']:.2f} MB",
            f"  Gradient memory (est):   {memory_info['gradient_memory_MB']:.2f} MB",
            f"  Total memory (est):      {memory_info['total_memory_MB']:.2f} MB ({memory_info['total_memory_GB']:.2f} GB)",
        ]
    )

    if training_info:
        lines.extend(
            [
                "",
                "[4] Training / 训练开销",
                f"  Total training time:     {training_info.get('total_training_time', 'N/A')}",
                f"  Num epochs:              {training_info.get('num_epochs', 'N/A')}",
                f"  Avg epoch time:          {training_info.get('avg_epoch_time', 'N/A')}",
                f"  Avg iter time:           {training_info.get('avg_iter_time_s', 'N/A')} s",
                f"  Training throughput:     {training_info.get('training_throughput', 'N/A')} img/s",
                f"  Peak GPU memory:         {training_info.get('peak_gpu_memory_MB', 'N/A')} MB",
            ]
        )

    lines.extend(["", "[5] Inference / 推理性能"])
    if inference_info and inference_info.get("success"):
        lines.extend(
            [
                f"  Input resolution:        {inference_info['input_height']} x {inference_info['input_width']}",
                f"  Latency (mean):          {inference_info['latency_ms_mean']:.2f} ms",
                f"  Latency (median):        {inference_info['latency_ms_median']:.2f} ms",
                f"  Latency (std):           {inference_info['latency_ms_std']:.2f} ms",
                f"  Latency (min/max):       {inference_info['latency_ms_min']:.2f} / {inference_info['latency_ms_max']:.2f} ms",
                f"  FPS:                     {inference_info['fps']:.2f}",
                f"  GPU peak memory:         {inference_info['gpu_memory_peak_MB']:.2f} MB",
                f"  Device:                  {inference_info['device']}",
            ]
        )
        if inference_info.get("eval_latency_ms_mean") is not None:
            lines.append(
                f"  Eval latency (dataset):  {inference_info['eval_latency_ms_mean']:.2f} ms/image"
            )
    else:
        err = inference_info.get("error") if inference_info else "not measured"
        lines.append(f"  Inference benchmark failed or skipped: {err}")

    lines.append("=" * 80)
    return "\n".join(lines)


def save_computational_cost_report(output_dir: str, report_text: str, report_dict: Dict[str, Any]) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, "computational_cost_report.txt")
    json_path = os.path.join(output_dir, "computational_cost_report.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    return txt_path, json_path


def log_computational_cost_report(logger, report_text: str) -> None:
    logger.info("\n" + report_text)


def analyze_model_complexity(model, cfg) -> Dict[str, Any]:
    """Static complexity analysis: params, FLOPs, estimated memory."""
    from util.complexity_analyzer import ComplexityAnalyzer

    batch_size, height, width = resolve_input_shape(cfg, model)
    input_shape = (batch_size, 3, height, width)
    analyzer = ComplexityAnalyzer(model, input_shape)
    return {
        "input_shape": input_shape,
        "params": analyzer.count_parameters(),
        "flops": analyzer.count_flops(),
        "memory": analyzer.get_memory_estimate(),
    }


def run_pretrain_complexity_analysis(model, cfg, accelerator, logger) -> Optional[Dict[str, Any]]:
    """Run params/FLOPs analysis before distributed wrapping."""
    if not accelerator.is_main_process:
        return None

    logger.info("\n" + "=" * 80)
    logger.info("Model complexity analysis (params / FLOPs / memory estimate)")
    logger.info("=" * 80)

    try:
        result = analyze_model_complexity(model, cfg)
        params_info = result["params"]
        flops_info = result["flops"]
        memory_info = result["memory"]
        input_shape = result["input_shape"]

        logger.info(f"Input shape: {input_shape}")
        logger.info(
            f"Params: total={params_info['total_M']:.2f}M, "
            f"trainable={params_info['trainable_M']:.2f}M"
        )
        logger.info(f"FLOPs: {flops_info['flops_G']:.2f} GFLOPs")
        logger.info(
            f"Memory (est): params={memory_info['params_memory_MB']:.1f}MB, "
            f"total={memory_info['total_memory_MB']:.1f}MB"
        )
        return result
    except Exception as exc:
        logger.warning(f"Complexity analysis failed: {exc}")
        return None


def run_inference_benchmark(model, cfg, accelerator, logger, warmup: int, num_runs: int) -> Optional[Dict[str, Any]]:
    """Benchmark inference latency/FPS/GPU memory on the prepared model."""
    if not accelerator.is_main_process:
        return None

    _, height, width = resolve_input_shape(cfg, model)
    logger.info("\n" + "=" * 80)
    logger.info(f"Inference benchmark @ {height}x{width} (warmup={warmup}, runs={num_runs})")
    logger.info("=" * 80)

    eval_model = accelerator.unwrap_model(model)
    inference_info = benchmark_inference(
        eval_model,
        accelerator.device,
        height=height,
        width=width,
        warmup=warmup,
        num_runs=num_runs,
    )

    if inference_info["success"]:
        logger.info(
            f"Latency: {inference_info['latency_ms_mean']:.2f} ms "
            f"(±{inference_info['latency_ms_std']:.2f}), "
            f"FPS: {inference_info['fps']:.2f}, "
            f"GPU peak: {inference_info['gpu_memory_peak_MB']:.1f} MB"
        )
    else:
        logger.warning(f"Inference benchmark failed: {inference_info['error']}")

    return inference_info


def finalize_computational_cost_report(
    complexity_result,
    inference_info,
    training_info,
    cfg,
    accelerator,
    logger,
) -> None:
    """Save final computational cost report after training."""
    if not accelerator.is_main_process or complexity_result is None:
        return

    params_info = complexity_result["params"]
    flops_info = complexity_result["flops"]
    memory_info = complexity_result["memory"]
    input_shape = complexity_result["input_shape"]

    report_text = format_computational_cost_report(
        params_info=params_info,
        flops_info=flops_info,
        memory_info=memory_info,
        inference_info=inference_info,
        training_info=training_info,
        input_shape=input_shape,
    )
    log_computational_cost_report(logger, report_text)

    report_dict = {
        "params": params_info,
        "flops": flops_info,
        "memory_estimate": memory_info,
        "inference": inference_info,
        "training": training_info,
        "input_shape": list(input_shape),
    }
    txt_path, json_path = save_computational_cost_report(cfg.output_dir, report_text, report_dict)
    logger.info(f"Computational cost report saved: {txt_path}")
    logger.info(f"Computational cost JSON saved: {json_path}")

    try:
        accelerator.log(
            {
                "model/total_params_M": params_info["total_M"],
                "model/flops_G": flops_info["flops_G"],
                "model/inference_latency_ms": inference_info.get("latency_ms_mean", 0) if inference_info else 0,
                "model/inference_fps": inference_info.get("fps", 0) if inference_info else 0,
                "model/inference_gpu_peak_MB": inference_info.get("gpu_memory_peak_MB", 0) if inference_info else 0,
                "train/total_time_s": training_info.get("total_training_time_s", 0),
                "train/peak_gpu_memory_MB": training_info.get("peak_gpu_memory_MB", 0),
            },
            step=cfg.num_epochs,
        )
    except Exception as exc:
        logger.warning(f"Failed to log compute metrics to TensorBoard: {exc}")
