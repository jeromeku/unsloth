import json
from dataclasses import asdict
from functools import partial

import torch

from unsloth.utils.profiling import TorchProfiler, _run_and_format_categories


def fn(mark_region, dtype, device):
    mark_region("Initialization")
    x = torch.randn(10, 20, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(20, 100, dtype=dtype, device=device, requires_grad=True)

    mark_region("Matmul")
    out = x @ w
    mark_region("Backward")
    out.sum().backward()

    return out


def test_memory_prof():
    f = partial(fn, device="cuda", dtype=torch.float32)
    profiler = TorchProfiler(with_flops=False, out_dir="./mem_test", prefix="mem_test")
    _, out = _run_and_format_categories(profiler, f)
    torch.save(out, "./mem_test/out.pt")
    # with open("./mem_test/out.json", "w") as f:
    #     json.dump([asdict(o) for o in out], f)
    # # torch.save(raw_out, "./mem_test/raw_out.pt")
    # torch.save(out, "./mem_test/out.pt")


test_memory_prof()
