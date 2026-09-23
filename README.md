# DAK: Direct-Access-Enabled GPU Memory Offloading with Optimal Efficiency for LLM Inference

This repository contains the research code for the paper:

**DAK: Direct-Access-Enabled GPU Memory Offloading with Optimal Efficiency for LLM Inference**  
Paper: <https://arxiv.org/pdf/2604.26074>

DAK is an end-to-end GPU memory offloading framework for large language model
inference. Instead of following the conventional prefetch-based design, where
offloaded data is first moved back into local GPU HBM before computation, DAK
enables the GPU to directly access offloaded memory and stream data into shared
memory for kernel execution.

The implementation in this repository includes CUDA kernels, PyTorch extension
bindings, attention kernels, model integration utilities, benchmarks, and
evaluation notebooks used to study direct-access memory offloading for LLM
inference.

## Overview

LLM inference is often limited by GPU memory capacity and bandwidth. Offloading
weights or KV caches to a remote memory tier can increase the effective memory
capacity, but existing systems commonly rely on prefetching data into GPU HBM.
This introduces several inefficiencies:

- additional HBM traffic and capacity pressure;
- pipeline bubbles when data movement is not perfectly hidden;
- underutilization of aggregate bandwidth across local and remote memory tiers;
- read amplification when multiple consumers need overlapping data.

DAK addresses these bottlenecks through direct-access offloading. The key idea
is to allow GPU kernels to fetch offloaded weights and KV cache blocks directly
from the remote tier into GPU shared memory, avoiding unnecessary staging in HBM.
The paper shows that this design can better aggregate local and remote memory
bandwidth, and reports up to **3x** speedup on NVLink-C2C systems and **1.8x**
speedup on PCIe systems over prefetch-based offloading baselines.

## Key Features

- **Direct-access offloading** for LLM inference workloads.
- **CUDA/PyTorch extension** for offloaded linear and runtime primitives.
- **TMA-based data movement** targeting Hopper-class GPUs.
- **Attention** with host/device KV cache partitioning.
- **Model integration utilities** for replacing selected linear layers and
  controlling placement.
- **Benchmarks and notebooks** for weight offloading, KV cache offloading,
  offloading-ratio analysis, and ablation studies.

## Repository Layout

```text
.
├── app/python/              # Model integration and placement utilities
├── benchmark/               # Benchmark entry points and scripts
├── eval/                    # Evaluation notebooks
├── include/                 # CUDA headers for runtime and task kernels
├── split_attention/         # Attention for splitted KV cache
├── python/offload/          # Python package for the offload runtime
├── src/                     # CUDA runtime and PyTorch binding sources
├── Makefile                 # Root build targets
└── setup.py                 # PyTorch CUDA extension build
```

## Requirements

The code is designed for CUDA-enabled systems and currently targets
Hopper-class GPUs by default:

- NVIDIA GPU with CUDA support;
- CUDA toolkit with `nvcc`;
- PyTorch with CUDA support;
- Python development environment with `pip`;
- `transformers`, `numpy`, and other benchmark-time Python dependencies.

The default build flags target `sm_90a`. If you are using a different GPU
architecture, update the `-gencode` settings in `Makefile`, `setup.py`, and
`split_attention/setup.py` accordingly.

## Installation

From the repository root directory, build and install the main offload runtime:

```bash
bash setup.sh
make pyext
```

Then install the split attention extension:

```bash
cd split_attention
pip install -e .
```

The root `make pyext` target first builds `runtime.o` and then installs the
editable PyTorch extension package. The `split_attention` package is installed
separately because it provides a dedicated attention kernel and Python
registration interface.

## Basic Test

To verify the installation, run from the repository root:

```bash
python test_install.py
```

The script checks that PyTorch can see a CUDA GPU and imports both compiled
extensions (`offload.runtime` and `opt_attention._C`). On success it prints
something like the following (versions, GPU name, and paths will differ):

```
torch 2.x.x, CUDA 13.0
GPU: NVIDIA GH200 480GB
offload: OK (.../offload/__init__.py)
opt_attention: OK (.../opt_attention/__init__.py)
Installation test PASSED
```

If a check fails, the script stops with an error. `CUDA is not available`
means PyTorch cannot see the GPU or driver. An `ImportError` means the
corresponding extension was not built or installed.

## Usage

The main Python modules are:

- `offload.runtime`: low-level runtime extension;
- `app.python.linear_replacement`: utilities for replacing model linear layers;
- `app.python.placement`: placement and offloading-ratio utilities;
- `split_attention`: split attention registration.

For attention experiments, register the attention implementation
before loading/running the model:

```python
import split_attention

split_attention.register(name="vdcores_opt")
```

The benchmark scripts under `benchmark/` provide examples for running OPT,
LLaMA, linear-layer, and attention experiments:

```bash
python benchmark/benchmark_opt.py
python benchmark/benchmark_opt_attn.py
python benchmark/benchmark_llama.py
python benchmark/benchmark_linear.py
```

Shell wrappers and experiment configurations are also provided under
`benchmark/script/`.

## Evaluation

The `eval/` directory contains notebooks for analyzing:

- offloading-ratio selection;
- offloading algorithm behavior;
- ablation studies.

These notebooks are intended to support the experimental analysis in the paper
and may require generated benchmark outputs.

## Notes

- This is research code and is optimized around the hardware/software
  assumptions used in the paper.
- Some kernels rely on Hopper-specific features such as Tensor Memory
  Accelerator support.
- For non-Hopper GPUs, the CUDA architecture flags and kernel assumptions may
  need to be adjusted.

## Citation

If you use this code or find the paper useful, please cite:

```bibtex
@article{lin2026dak,
  title={DAK: Direct-Access-Enabled GPU Memory Offloading with Optimal Efficiency for LLM Inference},
  author={Lin, Shouxu and Guo, Zhiyuan and Lin, Jiaxin},
  journal={arXiv preprint arXiv:2604.26074},
  year={2026}
}
```
