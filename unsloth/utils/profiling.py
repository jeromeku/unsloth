import dataclasses
import json
import os
import types
from datetime import datetime
from functools import partial

import pandas as pd
import torch
from torch.profiler._memory_profiler import (
    MemoryProfileTimeline,
)
from transformers.trainer_callback import ProgressCallback, TrainerCallback
from transformers.trainer_pt_utils import _secs2timedelta
from xformers.profiler.device_limits import get_device_limits

from .profiling_analyzer import AnalyzedTrace

# Prints filename and line number when logging
LOG_FORMAT_STR = (
    "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
)

TRAINER_PERF_ARGS = {
    "skip_memory_metrics": False,
    "include_num_input_tokens_seen": True,
    "include_tokens_per_second": True,
}


class MetricsCallBack(TrainerCallback):
    def __init__(self, name, verbose=False):
        self.name = name
        self.verbose = verbose

    def metrics_format(self, metrics):
        """
        Reformat Trainer metrics values to a human-readable format

        Args:
            metrics (`Dict[str, float]`):
                The metrics returned from train/evaluate/predict

        Returns:
            metrics (`Dict[str, float]`): The reformatted metrics
        """

        metrics_copy = metrics.copy()
        for k, v in metrics_copy.items():
            if "_mem_" in k:
                metrics_copy[k] = f"{ v >> 20 }MB"
            elif "_runtime" in k:
                metrics_copy[k] = _secs2timedelta(v)
            elif k == "total_flos":
                metrics_copy[k] = f"{ int(v) >> 30 }GF"
            elif isinstance(metrics_copy[k], float):
                metrics_copy[k] = round(v, 4)

        return metrics_copy

    def save_state(self, output_dir, state, append_step=False):
        # Format metrics (last entry of log_history)
        log_history = state.log_history
        metrics = self.metrics_format(log_history[-1])
        log_history[-1] = metrics
        state.log_history = log_history

        # Save state
        json_string = (
            json.dumps(dataclasses.asdict(state), indent=2, sort_keys=True) + "\n"
        )

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        step = "-" + str(state.global_step) if append_step else ""
        json_path = os.path.join(output_dir, f"{date_str}-{self.name}{step}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.verbose:
            logs_formatted = self.metrics_format(logs)
            k_width = max(len(str(x)) for x in logs_formatted.keys())
            v_width = max(len(str(x)) for x in logs_formatted.values())
            print("Global Step: ", state.global_step)
            for key in sorted(logs_formatted.keys()):
                print(f"  {key: <{k_width}} = {logs_formatted[key]:>{v_width}}")
        else:
            return

    def on_train_end(self, args, state, control, **kwargs):
        self.save_state(args.output_dir, state)


#        super().on_train_end(args, state, control, **kwargs)


class CudaProfilerCtx:
    def __enter__(self):
        print("Starting cuda profiler")
        torch.cuda.cudart().cudaProfilerStart()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        print("Stopping cuda profiler")
        torch.cuda.cudart().cudaProfilerStop()
        if exc_type is not None:
            print(f"Exception occurred: {exc_type}, {exc_value}")
        # Return True to suppress the exception
        return True

    def start(self):
        self.__enter__()

    def stop(self):
        self.__exit__(None, None, None)

    def step(self):
        pass


TIME_FORMAT_STR: str = "%m_%d"


def analyze_trace(prof):
    results = AnalyzedTrace.from_profile(prof.profiler.kineto_results.events())
    dtype_to_flops = get_device_limits(torch.device("cuda"))
    hw_flops = {}
    if dtype_to_flops is not None:
        for dtype, flops in dtype_to_flops.items():
            hw_flops[dtype] = flops
    total_flops = 0.0
    total_hfu = 0.0
    total_mfu = 0.0
    for dtype in results.operations_per_dtype_fw.keys():
        total_flops += results.compute_num_ops(dtype) / results.total_time_s
        total_hfu += results.compute_hfu(hw_flops)
        total_mfu += results.compute_mfu(hw_flops)

    hw_flops.update(("Step time (ms)", int(results.total_time_s * 1000)))
    hw_flops.update(("TFlops", total_flops / (1000**4)))
    hw_flops.update(("HFU", total_hfu))
    hw_flops.update(("MFU", total_mfu))
    return hw_flops


def trace_handler(
    prof: torch.profiler.profile,
    group_by_stack: int = 0,
    group_by_input_shapes: bool = False,
    with_stack: bool = True,
    with_flops: bool = True,
    prefix="",
    out_dir="./torch_profile",
    export_events=True,
    export_trace=True,
    export_memory_timeline=False,
    time_fmt_str: str = TIME_FORMAT_STR,
):
    # Prefix for file names.
    timestamp = datetime.now().strftime(time_fmt_str)
    file_prefix = os.path.join(out_dir, f"{prefix}-{timestamp}")

    if export_events:
        evt_list = prof.key_averages(
            group_by_stack_n=group_by_stack, group_by_input_shape=group_by_input_shapes
        )
        torch.save(evt_list, f"{file_prefix}-key_averages.pt")
        torch.save(prof.profiler.function_events, f"{file_prefix}-events.pt")
    # Construct the trace file.
    if export_trace:
        prof.export_chrome_trace(f"{file_prefix}-chrome-trace.json")

    # Construct the memory timeline file.
    if export_memory_timeline:
        prof.export_memory_timeline(
            f"{file_prefix}-memory-timeline.html", device="cuda:0"
        )
        prof.export_memory_timeline(
            f"{file_prefix}-memory-timeline.json", device="cuda:0"
        )
        mem_tl = MemoryProfileTimeline(prof._memory_profile())
        mem_tl.export_memory_timeline_raw(
            f"{file_prefix}-memory-timeline_raw.json", device_str="cuda:0"
        )
    if with_stack:
        prof.export_stacks(f"{file_prefix}-stacks.txt")

    if with_flops:
        analyze_traces(prof)
    print(
        prof.key_averages(
            group_by_input_shape=group_by_input_shapes, group_by_stack_n=group_by_stack
        ).table(sort_by="self_cuda_time_total", row_limit=-1)
    )


class TorchProfiler:
    def __init__(
        self,
        with_stack=True,
        with_flops=True,
        with_modules=False,
        record_shapes=True,
        export_events=True,
        export_trace=True,
        export_memory_timeline=True,
        group_by_stack=0,
        group_by_input_shapes=False,
        prefix="",
        out_dir=None,
        overwrite_existing=True,
        wait=0,
        warmup=0,
        active=5,
        repeat=1,
    ):
        if os.path.exists(out_dir) and overwrite_existing:
            import shutil

            shutil.rmtree(out_dir)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if group_by_input_shapes and not record_shapes:
            print(
                "group_by_input_shapes can only be True when record_shapes is True, setting record_shapes to True"
            )
            record_shapes = True
        callback = partial(
            trace_handler,
            export_events=export_events,
            export_trace=export_trace,
            export_memory_timeline=export_memory_timeline,
            group_by_input_shapes=group_by_input_shapes,
            group_by_stack=group_by_stack,
            with_stack=with_stack,
            with_flops=with_flops,
            prefix=prefix,
            out_dir=out_dir,
        )

        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=repeat
            ),
            record_shapes=record_shapes,
            profile_memory=export_memory_timeline,
            with_stack=with_stack,
            on_trace_ready=callback,
            with_flops=with_flops,
            with_modules=with_modules,
            experimental_config=torch._C._profiler._ExperimentalConfig(
                verbose=True,
                profiler_measure_per_kernel=True,
                enable_cuda_sync_events=True,
            )
            if with_stack
            else None,
        )

    def __enter__(self):
        return self.profiler.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.__exit__(exc_type, exc_val, exc_tb)

    def step(self):
        self.profiler.step()

    def start(self):
        self.profiler.start()

    def stop(self):
        self.profiler.stop()


_PERF_COLUMNS = [
    "key",
    "count",
    "cpu_children",
    "cpu_parent",
    "self_device_time_total",
    "cuda_time",
    "flops",
    "self_cpu_time",
    "self_cpu_time_total",
    "cpu_time",
    "cpu_time_total" "self_device_memory_usage",
    "device_memory_usage",
    "self_cpu_memory_usage",
    "cpu_memory_usage",
]
PERF_COLS_SELECT = [
    "key",
    "cpu_parent",
    "cpu_children",
    # "self_cpu_time",
    # "self_cpu_time_total",
    "cpu_time",
    "cpu_time_total",
    "cuda_time",
    "self_device_time_total",
]


# cuda_time, cpu_time are avg times -- corresponds to CUDA time avg and CPU time avg in table() above
# "self" times is not meaningful for annotated regions, since they only have child regions
def is_function(obj):
    return isinstance(obj, types.FunctionType)


def is_method(obj):
    return isinstance(obj, types.MethodType)


def is_private(prop):
    return prop.startswith("_")


def should_exclude(obj, prop):
    return (
        is_function(getattr(obj, prop))
        or is_method(getattr(obj, prop))
        or is_private(prop)
    )


def _get_event_props(event: torch.autograd.profiler_util.FunctionEvent):
    props = [p for p in dir(event) if not should_exclude(event, p)]
    return props


def get_events_df(events: torch.autograd.profiler_util.EventList):
    event_props = _get_event_props(events[0])
    data = [{p: getattr(e, p) for p in event_props} for e in events]
    return pd.DataFrame(data)


def get_perf_df(events: torch.autograd.profiler_util.EventList, sort=True):
    df = get_events_df(events).filter(PERF_COLS_SELECT)
    if sort:
        df = df.sort_values(["cpu_time", "cuda_time"], ascending=False)
    return df


def pivot_df(
    df,
    id_cols: str | list[str],
    columns: str | list[str],
    values: str | list[str],
    column_order: list[str] = None,
    show: bool = True,
):
    df = df.pivot_table(
        index=id_cols,
        columns=columns,
        values=values,
    ).reset_index()
    if column_order is not None:
        df = df[column_order]
    if show:
        print(df.to_string(index=False))
    return df
