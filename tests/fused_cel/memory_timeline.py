# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for torch profiler."""

import functools
import logging
import textwrap
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.cuda
from packaging import version
from torch._C._profiler import _EventType, _TensorMetadata
from torch.profiler import _memory_profiler, _utils
from torch.utils import _pytree as pytree

profile = functools.partial(
    torch.profiler.profile, record_shapes=True, profile_memory=True, with_stack=True
)
log = logging.getLogger(__name__)


# def export_memory_timeline_html(
#     prof: TorchProfile,
#     path: str,
#     device: Optional[str] = None,
#     figsize=(20, 12),
#     title=None,
#     yxis_step_size: float = 1.0,
#     return_fig: bool = False,
# ) -> Optional[Union[None, Any]]:
#     """Exports a memory timeline to an HTML file. Similar to the PyTorch plotting function, but with adjusted axis tickers and grids."""
#     if version.parse(torch.__version__) <= version.parse("2.1.0.dev"):
#         log.warning(
#             "export_memory_timeline_html failed because memory timeline is supported after PyTorch 2.1.0."
#         )
#         return

#     from torch.profiler._memory_profiler import (
#         _CATEGORY_TO_COLORS,
#         _CATEGORY_TO_INDEX,
#         MemoryProfileTimeline,
#     )

#     # Default to device 0, if unset. Fallback on cpu.
#     if device is None and prof.use_device and prof.use_device != "cuda":
#         device = prof.use_device + ":0"

#     if device is None:
#         device = "cuda:0" if torch.cuda.is_available() else "cpu"

#     # Construct the memory timeline plot data
#     mem_tl = MemoryProfileTimeline(prof._memory_profile())

#     # Check if user has matplotlib installed, return gracefully if not.
#     matplotlib_spec = importlib.util.find_spec("matplotlib")
#     if matplotlib_spec is None:
#         log.warning(
#             "export_memory_timeline_html failed because matplotlib was not found."
#         )
#         return
#     import matplotlib.pyplot as plt

#     mt = mem_tl._coalesce_timeline(device)
#     times, sizes = np.array(mt[0]), np.array(mt[1])
#     stacked = np.cumsum(sizes, axis=1) / 1024**3
#     max_memory_allocated = torch.cuda.max_memory_allocated()
#     max_memory_reserved = torch.cuda.max_memory_reserved()

#     # Plot memory timeline as stacked data
#     fig = plt.figure(figsize=figsize, dpi=80)
#     axes = fig.gca()
#     for category, color in _CATEGORY_TO_COLORS.items():
#         i = _CATEGORY_TO_INDEX[category]
#         axes.fill_between(
#             times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7
#         )
#     fig.legend(["Unknown" if i is None else i.name for i in _CATEGORY_TO_COLORS])
#     axes.set_xlabel("Time (us)")
#     axes.set_ylabel("Memory (GB)")
#     _, end = axes.get_ylim()
#     axes.grid(True)
#     axes.set_yticks(np.arange(0, end, yxis_step_size))
#     title = "\n\n".join(
#         ([title] if title else [])
#         + [
#             f"Max memory allocated: {max_memory_allocated/(10**9):.2f} GB \n"
#             f"Max memory reserved: {max_memory_reserved/(10**9):.2f} GB",
#         ]
#     )
#     axes.set_title(title)

#     if return_fig:
#         return fig

#     # Embed the memory timeline image into the HTML file
#     tmpfile = NamedTemporaryFile("wb", suffix=".png", delete=False)
#     tmpfile.close()
#     fig.savefig(tmpfile.name, format="png")

#     with open(tmpfile.name, "rb") as tmp:
#         encoded = b64encode(tmp.read()).decode("utf-8")
#         html = f"""<html>
#                 <head><meta charset="utf-8" /><title>GPU Memory Timeline HTML</title></head>
#                 <body>
#                 <img src='data:image/png;base64,{encoded}'>
#                 </body>
#                 </html>"""

#         with open(path, "w") as f:
#             f.write(html)
#     log.debug("Memory timeline exported to %s.", path)
#     remove(tmpfile.name)


def _run_and_format_data_flow(
    inputs: Dict[str, torch.Tensor],
    f: Callable[..., Optional[Dict[str, torch.Tensor]]],
    indent: int = 12,
) -> str:
    with profile() as prof:
        outputs = f(**inputs) or {}
        gc.collect()

    memory_profile = prof._memory_profile()
    graph = memory_profile._data_flow_graph
    storage_to_id = {key.storage.ptr: key.id for key in graph._active_version}

    lines: List[str] = []
    for name, t in it.chain(inputs.items(), outputs.items()):
        lines.append(f"{name + ':':<8} T{storage_to_id[t.storage().data_ptr()]}")
        if t.grad is not None:
            grad_id = storage_to_id[t.grad.storage().data_ptr()]
            lines.append(f"{name + '.grad:':<9} T{grad_id}")

    if lines:
        lines.append("")

    for node in graph.flow_nodes:
        destroyed = {k for k, v in node._edges.items() if v.is_deletion}

        inputs: List[str] = []
        for key, (_, v) in node.inputs.items():
            inputs.append(f"T{key.id}(v{v}{'*' if key in destroyed else ''})")

        outputs = [f"T{key.id}(v{v})" for key, v in node.outputs.items()]
        if inputs or outputs:
            event_name = node._event.name.replace("torch::autograd::", "")
            lines.append(
                f"{event_name:<25} {', '.join(inputs):<15}  ->  {', '.join(outputs)}"
            )

    return textwrap.indent("\n".join([l.rstrip() for l in lines]), " " * indent)


class RecordInputOutputDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        self.results = []

    def mark_region(self, name: str):
        self.results.append((name, (), ()))

    @staticmethod
    def flat_ids(args):
        flat_args = pytree.tree_leaves(args)
        return tuple(
            (t._cdata, t.storage().data_ptr())
            for t in flat_args
            if isinstance(t, torch.Tensor) and t.storage()
        )

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        args = args or []
        kwargs = kwargs or {}
        flat_inputs = self.flat_ids(args) + self.flat_ids(kwargs)
        out = func(*args, **kwargs)
        flat_outputs = self.flat_ids(out)
        if (
            flat_inputs or flat_outputs
        ) and "_record_function_enter" not in func.name():
            self.results.append((func.name(), flat_inputs, flat_outputs))
        return out


def _run_and_format_categories(fn, indent=12):
    """Generate summary of assigned categories for expecttest."""

    # Use `__torch_dispatch__` to collect ground truth.
    with RecordInputOutputDispatchMode() as record_ops, profile() as prof:
        fn(lambda name: record_ops.mark_region(f"-- {name} ".ljust(105, "-")))

    memory_profile = prof._memory_profile()
    ptr_pair_to_key: Dict[Tuple[int, int], _memory_profiler.TensorKey] = {}
    snapshot = memory_profile._category_snapshot()

    # Build map from observed live Tensors to the memory profiler's
    # TensorKey representation.
    for op in memory_profile._op_tree.dfs():
        if op.typed[0] == _EventType.TorchOp:
            inputs = pytree.tree_leaves(op.typed[1].inputs)
            for t in (i for i in inputs if isinstance(i, _TensorMetadata)):
                key = _memory_profiler.TensorKey.from_tensor(t)
                if key:
                    ptr_pair_to_key[(t.impl_ptr, t.storage_data_ptr)] = key

    def format_categories(ptr_pair: int):
        target_key = ptr_pair_to_key.get(ptr_pair, None)
        if target_key is None:
            return "???"

        matches = tuple(
            (version, category.name if category else "???")
            for (key, version), category in snapshot.items()
            if key == target_key
        )
        assert matches, "Failed to lookup Tensor"

        # Deduplicate version bumps which don't change the category.
        categories = [matches[0][1]]
        for _, category in matches:
            if category != categories[-1]:
                categories.append(category)

        return f"{target_key.storage.allocation_id} ({','.join(categories)})"

    out: List[str] = []
    for name, inputs, outputs in record_ops.results:
        if inputs or outputs:
            # PyTorch ops
            inputs_str = ", ".join(format_categories(i) for i in inputs)
            outputs_str = ", ".join(format_categories(i) for i in outputs)
            out.append(f"{name:<40} {inputs_str:<45} -> {outputs_str}")

        else:
            # Marked regions.
            out.append(f"\n{name}")

    return textwrap.indent("\n".join(out), " " * indent), memory_profile


def test_categories_e2e_simple_fwd_bwd() -> None:
    w0 = torch.ones((1,), requires_grad=True)
    w1 = torch.ones((1,), requires_grad=True)

    def step_fn(mark_region):
        x = torch.ones((2, 5), dtype=torch.float32, requires_grad=True)
        x2 = torch.ones((5, 10), dtype=torch.float32, requires_grad=True)
        # targets = torch.ones((2,), dtype=torch.float32)

        mark_region("Forward")
        y = x @ x2
        z = y.sum()

        mark_region("Backward")
        z.backward()
        # y = x + targets
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(y, targets)

        # mark_region("Backward")
        # loss.backward()

    out, mem_prof = _run_and_format_categories(step_fn)
    print(out)
    size_map_str = "\n".join(f"{k}: {v}" for k, v in mem_prof._size_map._values.items())
    print(size_map_str)
    print()


test_categories_e2e_simple_fwd_bwd()
s = """\
            aten::ones                                                                             -> 1 (INPUT)
            aten::ones                                                                             -> 2 (INPUT)

            -- Forward & loss ---------------------------------------------------------------------------------------
            aten::mul.Tensor                         1 (INPUT), 3 (INPUT)                          -> 4 (INPUT)
            aten::mul.Tensor                         1 (INPUT), 5 (INPUT)                          -> 6 (INPUT)
            aten::cat                                4 (INPUT), 6 (INPUT)                          -> 7 (INPUT)
            aten::binary_cross_entropy_with_logits   7 (INPUT), 2 (INPUT)                          -> 11 (INPUT)

            -- Backward ---------------------------------------------------------------------------------------------
            aten::ones_like                          11 (INPUT)                                    -> 14 (INPUT)
            aten::sigmoid                            7 (INPUT)                                     -> 15 (TEMPORARY)
            aten::sub.Tensor                         15 (TEMPORARY), 2 (INPUT)                     -> 16 (TEMPORARY)
            aten::mul.Tensor                         16 (TEMPORARY), 14 (INPUT)                    -> 17 (AUTOGRAD_DETAIL)
            aten::div_.Scalar                        17 (AUTOGRAD_DETAIL)                          -> 17 (AUTOGRAD_DETAIL)
            aten::slice.Tensor                       17 (AUTOGRAD_DETAIL)                          -> 17 (AUTOGRAD_DETAIL)
            aten::slice.Tensor                       17 (AUTOGRAD_DETAIL)                          -> 17 (AUTOGRAD_DETAIL)
            aten::mul.Tensor                         17 (AUTOGRAD_DETAIL), 1 (INPUT)               -> 20 (AUTOGRAD_DETAIL)
            aten::sum.dim_IntList                    20 (AUTOGRAD_DETAIL)                          -> 21 (GRADIENT)
            aten::view                               21 (GRADIENT)                                 -> 21 (GRADIENT)
            aten::detach                             21 (GRADIENT)                                 -> 21 (GRADIENT)
            aten::detach                             21 (GRADIENT)                                 -> ???
            aten::mul.Tensor                         17 (AUTOGRAD_DETAIL), 1 (INPUT)               -> 22 (AUTOGRAD_DETAIL)
            aten::sum.dim_IntList                    22 (AUTOGRAD_DETAIL)                          -> 23 (GRADIENT)
            aten::view                               23 (GRADIENT)                                 -> 23 (GRADIENT)
            aten::detach                             23 (GRADIENT)                                 -> 23 (GRADIENT)
            aten::detach                             23 (GRADIENT)                                 -> ???"""
