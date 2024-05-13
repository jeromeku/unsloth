import os
import types
from datetime import datetime
from functools import partial

import pandas as pd
import torch
import torch.autograd.profiler_util
from memory_timeline import export_memory_timeline_html as composer_memory_timeline
from tabulate import tabulate
from torch.autograd.profiler import record_function
from torch.cuda.nvtx import range as nvtx_range
from torch.profiler._memory_profiler import (
    _CATEGORY_TO_COLORS,
    _CATEGORY_TO_INDEX,
    MemoryProfileTimeline,
)


class FakeProfiler:
    def step(self):
        pass


# from torch.cuda.nvtx import range_pop, range_push

TIME_FORMAT_STR: str = "%m_%d"
PROFILE_DIR = "./profiles"


def trace_handler(
    prof: torch.profiler.profile,
    group_by_stack: int = 5,
    group_by_input_shapes: bool = False,
    prefix="",
    out_dir=None,
    export_events=False,
    export_trace=True,
    export_memory_timeline=False,
):
    # Prefix for file names.
    out_dir = out_dir or PROFILE_DIR
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = os.path.join(out_dir, f"{prefix}-{timestamp}")

    if export_events:
        evt_list = prof.key_averages(
            group_by_stack_n=group_by_stack, group_by_input_shape=group_by_input_shapes
        )
        torch.save(evt_list, f"{file_prefix}-key_averages.pt")

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
        composer_memory_timeline(
            prof, f"{file_prefix}-composer-memory-timeline.html", device="cuda:0"
        )
        mem_tl = MemoryProfileTimeline(prof._memory_profile())
        mem_tl.export_memory_timeline_raw(
            f"{file_prefix}-memory-timeline_raw.json", device_str="cuda:0"
        )

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))


def get_torch_profiler(
    name,
    with_stack=True,
    with_flops=True,
    with_modules=False,
    record_shapes=True,
    export_events=False,
    export_trace=True,
    export_memory_timeline=True,
    out_dir=None,
    warmup=1,
    active=5,
):
    if os.path.exists(out_dir):
        import shutil

        shutil.rmtree(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    callback = partial(
        trace_handler,
        prefix=name,
        out_dir=out_dir,
        group_by_input_shapes=record_shapes,
        group_by_stack=5 if export_events else None,
        export_events=export_events,
        export_trace=export_trace,
        export_memory_timeline=export_memory_timeline,
    )
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=record_shapes,
        with_stack=with_stack,
        with_flops=with_flops,
        with_modules=with_modules,
        profile_memory=export_memory_timeline,
        schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=active),
        on_trace_ready=callback,
    )
    return profiler


class TorchProfiler:
    def __init__(
        self,
        with_stack=True,
        with_flops=True,
        with_modules=False,
        record_shapes=True,
        export_events=False,
        export_trace=True,
        export_memory_timeline=True,
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

        callback = partial(
            trace_handler,
            export_events=export_events,
            export_trace=export_trace,
            export_memory_timeline=export_memory_timeline,
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


def ref_cel(hidden_states, lm_head_weight, labels):
    vocab_size = lm_head_weight.shape[0]
    with record_function("##logits"):
        logits = hidden_states @ lm_head_weight.T
        logits = logits.float()

    with record_function("##shift_logits"):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
    with record_function("##loss_fct"):
        loss_fct = torch.nn.CrossEntropyLoss()
        with record_function("##shift_logits_view"):
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
        with record_function("##loss_calc"):
            loss = loss_fct(shift_logits, shift_labels)

    return loss


def run_cel(fn, hidden_states, lm_head_weight, labels, **kwargs):
    loss = fn(hidden_states, lm_head_weight, labels, **kwargs)
    # dX, dW = torch.autograd.grad(loss, [hidden_states, lm_head_weight])
    return loss  # , dX, dW


def init(bs, seqlen, hidden_size, vocab_size, dtype):
    with record_function("##initialization"):
        hidden_states = torch.randn(
            bs, seqlen, hidden_size, dtype=dtype, device="cuda", requires_grad=True
        )

        lm_head_weight = torch.randn(
            (vocab_size, hidden_size), dtype=dtype, device="cuda", requires_grad=True
        )

        # Input ids aren't actually used, but we need to pass them to the model
        labels = torch.randint(0, vocab_size, size=(bs, seqlen), device="cuda")
    with record_function("##create_optimizer"):
        optimizer = torch.optim.AdamW([lm_head_weight])
    # Reference loss, dX, dW where dX is the gradients wrt to the hidden states and dW is the gradients wrt to the LM head weight
    return hidden_states, lm_head_weight, labels, optimizer


def run_step(hidden_states, lm_head_weight, labels, optimizer):
    with record_function("##train_step"):
        with record_function("##forward"):
            loss = run_cel(ref_cel, hidden_states, lm_head_weight, labels)
        with record_function("##backward"):
            loss.backward()
        with record_function("##optimizer_step"):
            optimizer.step()
        with record_function("##optimizer_zero_grad"):
            optimizer.zero_grad()


def main():
    bs = 1
    seqlen = 256
    hidden_size = 128
    vocab_size = 32000
    dtype = torch.float32
    # Warm up
    hidden_states, lm_head_weight, labels, optimizer = init(
        bs, seqlen, hidden_size, vocab_size, dtype
    )

    for _ in range(5):
        run_step(hidden_states, lm_head_weight, labels, optimizer)

    # Clean up
    del hidden_states, lm_head_weight, labels, optimizer
    import gc

    torch.cuda.memory.empty_cache()
    gc.collect()

    profiler = TorchProfiler(
        with_stack=True,
        with_flops=True,
        with_modules=False,
        record_shapes=True,
        export_events=False,
        export_trace=True,
        export_memory_timeline=True,
        prefix="ref",
        out_dir="profs",
    )
    profiler.start()
    hidden_states, lm_head_weight, labels, optimizer = init(
        bs, seqlen, hidden_size, vocab_size, dtype
    )
    for _ in range(10):
        run_step(hidden_states, lm_head_weight, labels, optimizer)
        profiler.step()

    profiler.stop()


if __name__ == "__main__":
    main()
