import argparse
import datetime
import os
import pprint
import re
import time

import accelerate
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import ProjectConfiguration
from torch.utils import data

from util.collate_fn import collate_fn
from util.compute_metrics import (
    finalize_computational_cost_report,
    get_gpu_memory_mb,
    reset_peak_gpu_memory,
    run_inference_benchmark,
    run_pretrain_complexity_analysis,
    summarize_training_epoch,
)
from util.engine import evaluate_acc, train_one_epoch_acc
from util.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from util.lazy_load import Config
from util.misc import default_setup, encode_labels, fixed_generator, seed_worker
from util.utils import HighestCheckpoint, load_checkpoint, load_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config-file", default="configs/train_config.py")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--accumulate-steps", type=int, default=1, help="Steps to accumulate gradients"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--use-deterministic-algorithms", action="store_true")
    parser.add_argument(
        "--report-compute-cost",
        action="store_true",
        help="Report params, FLOPs, training time, inference latency/FPS, and GPU memory",
    )
    parser.add_argument(
        "--analyze-complexity",
        action="store_true",
        help="Alias of --report-compute-cost for backward compatibility",
    )
    parser.add_argument(
        "--benchmark-inference",
        action="store_true",
        help="Run inference speed benchmark (enabled with --report-compute-cost)",
    )
    parser.add_argument(
        "--inference-warmup",
        type=int,
        default=10,
        help="Warmup iterations for inference benchmark",
    )
    parser.add_argument(
        "--inference-runs",
        type=int,
        default=50,
        help="Benchmark iterations for inference latency/FPS",
    )
    dynamo_backend = ["no", "eager", "aot_eager", "inductor", "aot_ts_nvfuser", "nvprims_nvfuser"]
    dynamo_backend += ["cudagraphs", "ofi", "fx2trt", "onnxrt", "tensorrt", "ipex", "tvm"]
    parser.add_argument(
        "--dynamo-backend",
        type=str,
        default="no",
        choices=dynamo_backend,
        help="""
        Set to one of the possible dynamo backends to optimize the training with torch dynamo.
        See https://pytorch.org/docs/stable/torch.compiler.html and
        https://huggingface.co/docs/accelerate/main/en/package_reference/utilities#accelerate.utils.DynamoBackend
        """,
    )

    args = parser.parse_args()
    args.report_compute_cost = args.report_compute_cost or args.analyze_complexity
    return args


def update_checkpoint_path(cfg: Config):
    weight_path = getattr(cfg, "resume_from_checkpoint", None)
    if weight_path is not None and os.path.isdir(weight_path):
        cfg.output_dir = weight_path
    elif getattr(cfg, "output_dir", None) is None:
        accelerate.utils.wait_for_everyone()
        cfg.output_dir = os.path.join(
            "checkpoints",
            os.path.basename(cfg.model_path).split(".")[0],
            "train",
            datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"),
        )

    if weight_path is not None and os.path.isdir(weight_path):
        if "checkpoints" in os.listdir(cfg.resume_from_checkpoint):
            output_dir = os.path.join(cfg.resume_from_checkpoint, "checkpoints")
            folders = [os.path.join(output_dir, folder) for folder in os.listdir(output_dir)]
            folders.sort(
                key=lambda folder: list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[
                    0]
            )
            cfg.resume_from_checkpoint = folders[-1]
        else:
            cfg.resume_from_checkpoint = None

    return cfg


def train():
    args = parse_args()

    lazy_loads = ("lr_scheduler", "optimizer", "param_dicts")
    cfg = Config(file_path=args.config_file, partials=lazy_loads)
    cfg = update_checkpoint_path(cfg)

    project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, total_limit=5, automatic_checkpoint_naming=True
    )
    tensorboard_tracker = TensorBoardTracker(run_name="tf_log", logging_dir=cfg.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.find_unused_parameters)
    accelerator = Accelerator(
        log_with=tensorboard_tracker,
        project_config=project_config,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulate_steps,
        dynamo_backend=args.dynamo_backend,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[kwargs],
    )
    accelerator.init_trackers("det_train")
    default_setup(args, cfg, accelerator)
    logger = get_logger(os.path.basename(os.getcwd()) + "." + __name__)

    params = dict(num_workers=cfg.num_workers, collate_fn=collate_fn)
    params.update(dict(pin_memory=cfg.pin_memory, persistent_workers=True))
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        params.update({"worker_init_fn": seed_worker, "generator": fixed_generator()})

    group_ids = create_aspect_ratio_groups(cfg.train_dataset, k=3)
    train_batch_sampler = GroupedBatchSampler(
        data.RandomSampler(cfg.train_dataset), group_ids, cfg.batch_size
    )
    train_loader = data.DataLoader(cfg.train_dataset, batch_sampler=train_batch_sampler, **params)
    test_loader = data.DataLoader(cfg.test_dataset, 1, shuffle=False, **params)

    model = Config(cfg.model_path).model

    complexity_result = None
    if args.report_compute_cost:
        complexity_result = run_pretrain_complexity_analysis(model, cfg, accelerator, logger)

    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = cfg.optimizer(cfg.param_dicts(model))
    lr_scheduler = cfg.lr_scheduler(optimizer)

    weight_path = getattr(cfg, "resume_from_checkpoint", None)
    if weight_path is not None and os.path.isfile(weight_path):
        checkpoint = load_checkpoint(cfg.resume_from_checkpoint)
        load_state_dict(model, checkpoint)
        logger.info(f"load pretrained from {cfg.resume_from_checkpoint}")

    cat_ids = list(range(max(cfg.train_dataset.coco.cats.keys()) + 1))
    classes = tuple(cfg.train_dataset.coco.cats.get(c, {"name": "none"})["name"] for c in cat_ids)
    model.register_buffer("_classes_", torch.tensor(encode_labels(classes)))

    model, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, lr_scheduler
    )

    if weight_path is not None and os.path.isdir(weight_path):
        accelerator.load_state(cfg.resume_from_checkpoint)
        path = os.path.basename(cfg.resume_from_checkpoint)
        cfg.starting_epoch = int(path.split("_")[-1]) + 1
        accelerator.project_configuration.iteration = cfg.starting_epoch
        logger.info(f"resume training of {cfg.output_dir}, from {path}")
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("model parameters: {}".format(n_params))
        logger.info("optimizer: {}".format(optimizer))
        logger.info("lr_scheduler: {}".format(pprint.pformat(lr_scheduler.state_dict())))

    inference_info = None
    if args.report_compute_cost:
        inference_info = run_inference_benchmark(
            model,
            cfg,
            accelerator,
            logger,
            warmup=args.inference_warmup,
            num_runs=args.inference_runs,
        )

    if accelerator.is_main_process:
        label_file = os.path.join(cfg.output_dir, "label_names.txt")
        with open(label_file, "w") as f:
            caid_name = [f"{k} {v['name']}" for k, v in cfg.train_dataset.coco.cats.items()]
            caid_name = "\n".join(caid_name)
            f.write(caid_name)
        logger.info(f"Label names is saved to {label_file}")

    logger.info("Start training")
    reset_peak_gpu_memory()
    start_time = time.perf_counter()
    highest_checkpoint = HighestCheckpoint(accelerator, model)

    epoch_times = []
    epoch_train_stats = []
    peak_gpu_memory_mb = 0.0
    last_eval_latency_ms = None

    for epoch in range(cfg.starting_epoch, cfg.num_epochs):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch_acc(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epoch=epoch,
            print_freq=cfg.print_freq,
            max_grad_norm=cfg.max_norm,
            accelerator=accelerator,
        )
        lr_scheduler.step()

        epoch_time_s = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time_s)
        epoch_stats = summarize_training_epoch(train_metrics)
        epoch_stats["epoch_time_s"] = epoch_time_s
        epoch_train_stats.append(epoch_stats)

        mem = get_gpu_memory_mb()
        peak_gpu_memory_mb = max(peak_gpu_memory_mb, mem["max_allocated_MB"])

        accelerator.save_state(safe_serialization=False)
        logger.info("Start evaluation")
        coco_evaluator = evaluate_acc(model, test_loader, epoch, accelerator)
        eval_stats = getattr(coco_evaluator, "eval_stats", {})
        if eval_stats.get("eval_latency_ms_mean") is not None:
            last_eval_latency_ms = eval_stats["eval_latency_ms_mean"]
            if inference_info is not None:
                inference_info["eval_latency_ms_mean"] = last_eval_latency_ms

        cur_ap, cur_ap50 = coco_evaluator.coco_eval["bbox"].stats[:2]
        highest_checkpoint.update(ap=cur_ap, ap50=cur_ap50)

        if args.report_compute_cost:
            logger.info(
                f"Epoch {epoch} time: {datetime.timedelta(seconds=int(epoch_time_s))}, "
                f"iter_time: {epoch_stats.get('iter_time_avg_s', 0):.4f}s, "
                f"peak GPU mem: {peak_gpu_memory_mb:.0f} MB"
            )

    total_time_s = time.perf_counter() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time_s)))
    logger.info("Training time: {}".format(total_time))

    if args.report_compute_cost:
        num_epochs_ran = max(len(epoch_times), 1)
        avg_epoch_time_s = sum(epoch_times) / num_epochs_ran
        avg_iter_time_s = None
        training_throughput = None
        if epoch_train_stats:
            iter_times = [s.get("iter_time_avg_s") for s in epoch_train_stats if s.get("iter_time_avg_s")]
            if iter_times:
                avg_iter_time_s = sum(iter_times) / len(iter_times)
                training_throughput = f"{1.0 / avg_iter_time_s:.2f}" if avg_iter_time_s > 0 else "N/A"

        training_info = {
            "total_training_time": total_time,
            "total_training_time_s": total_time_s,
            "num_epochs": len(epoch_times),
            "avg_epoch_time": str(datetime.timedelta(seconds=int(avg_epoch_time_s))),
            "avg_epoch_time_s": avg_epoch_time_s,
            "avg_iter_time_s": f"{avg_iter_time_s:.4f}" if avg_iter_time_s is not None else "N/A",
            "training_throughput": training_throughput if training_throughput is not None else "N/A",
            "peak_gpu_memory_MB": f"{peak_gpu_memory_mb:.1f}",
            "eval_latency_ms_mean": last_eval_latency_ms,
        }
        finalize_computational_cost_report(
            complexity_result,
            inference_info,
            training_info,
            cfg,
            accelerator,
            logger,
        )

    accelerator.end_training()


if __name__ == "__main__":
    train()
