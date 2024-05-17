import gc
from functools import partial
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import GELU, Linear, Module


def get_memory_stats():
    print(torch.cuda.memory_summary())


def get_max_memory_reserved():
    mem = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"Peak reserved memory = {mem} GB.")
    return mem


def get_mem_stat_keys():
    mem_stats = torch.cuda.memory_stats_as_nested_dict()
    return mem_stats.keys()


def get_mem_stat(key):
    mem_stats = torch.cuda.memory_stats_as_nested_dict()
    try:
        return mem_stats[key]
    except:
        print(
            f"Key {key} not found in memory stats, run `get_mem_stat_keys()` to see all keys."
        )
        return None


def empty_cache():
    # Clean memory up first
    for _ in range(3):
        torch.cuda.empty_cache()
        gc.collect()
    pass


def mib_str(bytes: int) -> str:
    return f"{bytes/1024**2:.2f}MiB"


def pretty_memory_snapshot() -> str:
    # print(torch.cuda.memory_snapshot())

    return "\n".join(
        [
            f"""{m['address']}: {mib_str(m['allocated_size']).rjust(11)} alloc, {mib_str(m['total_size']).rjust(11)} total"""
            for m in torch.cuda.memory_snapshot()
        ]
    )


def pretty_mem(preamble: str, context: str, device_ix=0):
    alloc: int = torch.cuda.memory_allocated(device_ix)
    total: int = torch.cuda.memory_reserved(device_ix)
    reserved: int = total - alloc
    return f"{preamble}{context.rjust(20)} {mib_str(alloc).rjust(11)} alloc {mib_str(reserved).rjust(11)} reserved {mib_str(total).rjust(11)} total"


class TorchMemCtx:
    def __enter__(self):
        self.begin_alloc = torch.cuda.memory_allocated()
        self.begin_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_max_memory_allocated()
        return

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        end_alloc = torch.cuda.memory_allocated()
        peak_alloc = torch.cuda.max_memory_allocated()
        end_reserved = torch.cuda.memory_reserved()
        peak_reserved = torch.cuda.max_memory_reserved()
        print("Begin memory allocated:", self.begin_alloc)
        print(
            f"Memory allocated (end - begin): {mib_str(end_alloc - self.begin_alloc)}"
        )
        print(
            f"Peak memory allocated (max - begin): {mib_str(peak_alloc - self.begin_alloc)}"
        )

        print("Begin memory reserved:", self.begin_reserved)
        print(
            f"Memory reserved (end - begin): {mib_str(end_reserved - self.begin_reserved)}"
        )
        print(
            f"Peak memory reserved (max - begin): {mib_str(peak_reserved - self.begin_reserved)}"
        )


def memory_hook_fn(
    m: Module, inputs: Tuple[Tensor, ...], outputs: Tuple[Tensor, ...], msg: str = ""
) -> Optional[Tensor]:
    #  torch.cuda.synchronize()
    has_inputs = len(inputs) > 0
    has_outputs = len(outputs) > 0
    print(
        pretty_mem(
            "",
            f"{msg} {m.__class__.__name__} inputs {[x.shape for x in inputs if x is not None] if has_inputs else None} outputs {outputs[0].shape if outputs is not None else None}",
        )
    )  # {o[0].shape}:"))
    print(pretty_memory_snapshot())


def pre_hook_fn(
    model: Module, args, msg=""
):  # , o: Tuple[Tensor, ...]) -> Optional[Tensor]:
    """
    Record memory first forward / backwards pass

    Usage:
        model.layers[-1].register_full_backward_pre_hook(pre_hook_fn)

    """
    #  torch.cuda.synchronize()
    print(
        pretty_mem(
            "",
            f"{msg} {model.__class__.__name__} arg shapes: {[x.shape for x in args if x is not None]}",
        )
    )  # {o[0].shape}:"))
    print(pretty_memory_snapshot())


def add_fwd_bwd_memory_hooks(
    mod: Module, accepted_modules: set[Module], fwd=True, bwd=True, msg=""
) -> None:
    """
    Record memory after backwards pass for matching modules
    Usage:
        memory_hook = partial(add_memory_hooks, accepted_modules={Linear, GELU})
        model.apply(memory_hook)

    Args:
        mod (Module): _description_
    """
    if any(isinstance(mod, m) for m in accepted_modules):
        if fwd:
            mod.register_forward_hook(
                partial(
                    memory_hook_fn,
                    msg=msg if msg else "After fwd",
                )
            )
        if bwd:
            mod.register_full_backward_hook(
                partial(
                    memory_hook_fn,
                    msg=msg if msg else "After bwd",
                )
            )


def apply_memory_hooks(
    model: Module,
    accepted_modules: set[Module],
    fwd=True,
    bwd=True,
    before_first_fwd=False,
    before_first_bwd=True,
) -> None:
    apply_memory_hook = partial(
        add_fwd_bwd_memory_hooks, accepted_modules=accepted_modules, fwd=fwd, bwd=bwd
    )
    for m in model.modules():
        apply_memory_hook(m)
    if before_first_fwd:
        first_module = next(iter(model.modules()))
        hook_fn = partial(pre_hook_fn, msg="Before model forward")
        first_module.register_forward_pre_hook(hook_fn)
    if before_first_bwd:
        last_module = next(reversed(list(model.modules())))
        hook_fn = partial(pre_hook_fn, msg="Before model backward")
        last_module.register_full_backward_pre_hook(hook_fn)

    return model
