# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple, Union

import torch

# https://github.com/Lightning-AI/pytorch-lightning/src/lightning/fabric/utilities/throughput.py
_CUDA_FLOPS: Dict[str, Dict[Union[str, torch.dtype], float]] = {
    # Hopper
    # source: https://resources.nvidia.com/en-us-tensor-core
    "h100 nvl": {
        torch.float64: 67e12,
        torch.float32: 133.8e12,
        "tfloat32": 989.4e12,
        torch.bfloat16: 1978.8e12,
        torch.float16: 1978.8e12,
        torch.int8: 3957.8e12,
    },
    "h100 sxm": {
        torch.float64: 33.5e12,
        torch.float32: 66.9e12,
        "tfloat32": 494.7e12,
        torch.bfloat16: 989.4e12,
        torch.float16: 989.4e12,
        torch.int8: 1978.9e12,
    },
    "h100 pcie": {
        torch.float64: 25.6e12,
        torch.float32: 51.2e12,
        "tfloat32": 378e12,
        torch.bfloat16: 756e12,
        torch.float16: 756e12,
        torch.int8: 1513e12,
    },
    # Ada
    # source: https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf
    "rtx 4090": {
        torch.float32: 82.6e12,
        "tfloat32": 82.6e12,
        torch.bfloat16: 82.6e12,
        torch.float16: 82.6e12,
        torch.int8: 660.6e12,
        "int4": 1321.2e12,
    },
    "rtx 4080": {
        torch.float32: 48.7e12,
        "tfloat32": 48.7e12,
        torch.bfloat16: 48.7e12,
        torch.float16: 48.7e12,
        torch.int8: 389.9e12,
        "int4": 779.8e12,
    },
    "l4": {
        torch.float32: 30.3e12,
        "tfloat32": 60e12,
        torch.bfloat16: 121e12,
        torch.float16: 121e12,
        torch.int8: 242e12,
        "int4": 484e12,
    },
    "l40": {
        torch.float32: 90.5e12,
        "tfloat32": 90.5e12,
        torch.bfloat16: 181e12,
        torch.float16: 181e12,
        torch.int8: 362e12,
        "int4": 724e12,
    },
    # Ampere
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # sxm and pcie have same flop counts
    "a100": {
        torch.float64: 9.7e12,
        torch.float32: 19.5e12,
        "tfloat32": 156e12,
        torch.bfloat16: 312e12,
        torch.float16: 312e12,
        torch.int8: 624e12,
    },
    "a6000": {
        torch.float32: 38.7e12,
        "tfloat32": 77.4e12,
        torch.bfloat16: 38.7e12,
        torch.float16: 38.7e12,
        torch.int8: 309.7e12,
        "int4": 619.3e12,
    },
    "a40": {
        torch.float32: 37.4e12,
        "tfloat32": 74.8e12,
        torch.bfloat16: 37.4e12,
        torch.float16: 37.4e12,
        torch.int8: 299.3e12,
        "int4": 598.7e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
    "a10g": {
        torch.float32: 31.2e12,
        "tfloat32": 62.5e12,
        torch.bfloat16: 125e12,
        torch.float16: 125e12,
        torch.int8: 250e12,
        "int4": 500e12,
    },
    "rtx 3090 ti": {
        torch.float32: 40e12,
        "tfloat32": 40e12,
        torch.bfloat16: 40e12,
        torch.float16: 40e12,
        torch.int8: 320e12,
        "int4": 640e12,
    },
    "rtx 3090": {
        torch.float32: 35.6e12,
        "tfloat32": 35.6e12,
        torch.bfloat16: 35.6e12,
        torch.float16: 35.6e12,
        torch.int8: 284e12,
        "int4": 568e12,
    },
    "rtx 3080 ti": {
        torch.float32: 34.1e12,
        "tfloat32": 34.1e12,
        torch.bfloat16: 34.1e12,
        torch.float16: 34.1e12,
        torch.int8: 272.8e12,
        "int4": 546.6e12,
    },
    "rtx 3080": {
        torch.float32: 29.8e12,
        "tfloat32": 29.8e12,
        torch.bfloat16: 29.8e12,
        torch.float16: 29.8e12,
        torch.int8: 238e12,
        "int4": 476e12,
    },
    "rtx 3070": {
        torch.float32: 20.3e12,
        "tfloat32": 20.3e12,
        torch.bfloat16: 20.3e12,
        torch.float16: 20.3e12,
        torch.int8: 162.6e12,
        "int4": 325.2e12,
    },
    # Turing
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    # sxm and pcie have same flop counts
    "t4": {
        torch.float32: 8.1e12,
        torch.float16: 65e12,
        torch.int8: 130e12,
        "int4": 260e12,
    },
    # https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-5000-data-sheet-us-nvidia-704120-r4-web.pdf
    "quadro rtx 5000": {
        torch.float32: 11.2e12,
        torch.float16: 89.2e12,
    },
    "rtx 2080 super": {
        torch.float32: 11.2e12,
        torch.float16: 22.3e12,
        torch.int8: 178.4e12,
        "int4": 356.8e12,
    },
    "rtx 2080 ti": {
        torch.float32: 14.2e12,
        torch.float16: 28.5e12,
        torch.int8: 227.7e12,
        "int4": 455.4e12,
    },
    "rtx 2080": {
        torch.float32: 10.6e12,
        torch.float16: 21.2e12,
        torch.int8: 169.6e12,
        "int4": 339.1e12,
    },
    # https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
    "rtx 2070 super": {
        torch.float32: 9.1e12,
        torch.float16: 18.1e12,
        torch.int8: 145e12,
        "int4": 290e12,
    },
    "titan rtx": {
        torch.float32: 16.3e12,
        torch.float16: 32.6e12,
        torch.int8: 261e12,
        "int4": 522e12,
    },
    # Volta
    # source: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    "v100 sxm": {
        torch.float64: 7.8e12,
        torch.float32: 15.7e12,
        torch.float16: 125e12,
    },
    "v100 pcie": {
        torch.float64: 7e12,
        torch.float32: 14e12,
        torch.float16: 112e12,
    },
    "v100s pcie": {
        torch.float64: 8.2e12,
        torch.float32: 16.4e12,
        torch.float16: 130e12,
    },
}


def get_available_flops(device: torch.device = torch.device("cuda")):
    """Returns the available theoretical FLOPs.

    This is an optimistic upper limit that could only be achievable if only thick matmuls were run in a benchmark
    environment.

    """
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        chip = device_name.lower()
        if "h100" in chip:
            if "hbm3" in chip:
                chip = "h100 sxm"
            elif "nvl" in chip:
                chip = "h100 nvl"
            elif "pcie" in chip or "hbm2e" in chip:
                chip = "h100 pcie"
        elif "l4" in chip:
            chip = "l40" if "tesla" in chip else "l4"
        elif "geforce rtx" in chip:
            number = chip.split(" ")[3]
            extra = ""
            if "super" in chip:
                extra = " super"
            elif "ti" in chip:
                extra = " ti"
            chip = f"rtx {number}{extra}"
        elif "a6000" in chip:
            chip = "a6000"
        elif "a100" in chip:
            chip = "a100"
        elif "a40" in chip:
            chip = "a40"
        elif "a10g" in chip:
            chip = "a10g"
        elif "t4" in chip:
            chip = "t4"
        elif "quadro rtx 5000" in chip:
            chip = "quadro rtx 5000"
        elif "titan rtx" in chip:
            chip = "titan rtx"
        elif "v100-sxm" in chip:
            chip = "v100 sxm"
        elif "v100-pcie" in chip:
            chip = "v100 pcie"
        elif "v100s-pcie" in chip:
            chip = "v100s pcie"
        else:
            # the flops list is not exhaustive, return with a warning
            print(f"FLOPs not found for {device_name!r}")
            return None
        if chip not in _CUDA_FLOPS:
            # parsing is implemented but we don't have the stats
            print(f"FLOPs not found for {device_name!r}, chip is {chip!r}")
            return None
        dtype_to_flops = _CUDA_FLOPS[chip]
        # if dtype is torch.float32:
        #     major, _ = torch.cuda.get_device_capability(device)
        #     is_ampere_or_later = major >= 8

        #     if is_ampere_or_later and torch.get_float32_matmul_precision() != "highest":
        #         dtype = "tfloat32"

        # if dtype not in dtype_to_flops:
        #     # for example, T4 doesn't support bfloat16. it might also be that we are missing this dtype from the list
        #     print(f"{device_name!r} does not support {dtype}")
        #     return None
        return dtype_to_flops


@dataclass
class DeviceLimit:
    name: str = "default"  # pattern to match from `torch.cuda.get_device_name()`
    source: str = ""
    sm: Tuple[int, int] = (0, 0)
    # bytes/s
    gmem_bandwidth: float = math.inf
    # dtype -> TFlop/s
    gemm_tflops: Mapping[torch.dtype, float] = field(default_factory=dict)


# For f32, we assume we can use tf32
DEVICE_LIMITS: Tuple[DeviceLimit, ...] = (
    DeviceLimit(
        "H100",
        "https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet",  # noqa: E501
        sm=(9, 0),
        gmem_bandwidth=3.35 * (1024**4),  # NOTE: PCIe is 2 TB/s
        gemm_tflops={
            torch.float64: 67,
            # NOTE: NVIDIA gives all numbers "with 2:4 sparsity"
            # but we want the full GEMM numbers
            torch.float32: 989 // 2,
            torch.float16: 1979 // 2,
            torch.bfloat16: 1979 // 2,
            torch.int8: 3958 // 2,
        },
    ),
    DeviceLimit(
        "A100",
        "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf",  # noqa: E501
        sm=(8, 0),
        gmem_bandwidth=2 * (1024**4),  # NOTE: PCIe is 1.5 TB/s
        gemm_tflops={
            torch.float64: 19.5,
            torch.float32: 156,
            torch.float16: 312,
            torch.bfloat16: 312,
            torch.int8: 624,
        },
    ),
    DeviceLimit(
        "A30",
        "https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/products/a30-gpu/pdf/a30-datasheet.pdf",
        sm=(8, 0),
        gmem_bandwidth=933 * (1024**3),
        gemm_tflops={
            torch.float64: 10.3,
            torch.float32: 82,
            torch.float16: 165,
            torch.bfloat16: 165,
            torch.int8: 330,
        },
    ),
    DeviceLimit(
        "T4",
        "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf",
        sm=(7, 5),
        gmem_bandwidth=300 * (1024**3),
        gemm_tflops={
            torch.float32: 8.1,
            torch.float16: 65,
            torch.int8: 130,
        },
    ),
    # Assuming SXM2
    DeviceLimit(
        "V100",
        "https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf",
        sm=(7, 0),
        gmem_bandwidth=900 * (1024**3),
        gemm_tflops={
            torch.float64: 7.8,
            torch.float32: 15.7,
            torch.float16: 125,
        },
    ),
    DeviceLimit(
        "P100",
        "https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-datasheet.pdf",
        sm=(6, 0),
        gmem_bandwidth=732 * (1024**3),
        gemm_tflops={
            torch.float64: 5.3,
            torch.float32: 10.6,
            torch.float16: 21.2,
        },
    ),
)


def get_device_limits(device) -> DeviceLimit:
    """Currently only implemented for GPUs"""
    dtype_to_flops = get_available_flops()
    # if device is not None and device.type == "cuda":
    #     device_sm = torch.cuda.get_device_capability(device)
    #     device_name = torch.cuda.get_device_name(device)
    #     for lim in DEVICE_LIMITS:
    #         if lim.sm == device_sm:
    #             if lim.name in device_name:
    #                 return lim
    # return DeviceLimit()
    return dtype_to_flops
