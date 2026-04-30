# Benchmark Guide

This directory contains the benchmark entry points and scripts used to
reproduce the main end-to-end results in the DAK paper.

Run the commands below from the `benchmark/` directory after installing the
root offload runtime and the `split_attention` extension.

## Reproduce Paper Figures

### Figure 8

```bash
./script/opt/mem_bound.sh
```

### Figure 9

```bash
./script/opt/mix_bound.sh
./script/llama/mix_bound.sh
```

### Figure 10

```bash
./script/opt/mix_bound_uniform.sh
./script/llama/mix_bound_uniform.sh
```

## Notes on Kernel Configuration

The current kernel configuration, including `TILE_SIZE`, `CHUNK_SIZE`, and
`BUFFER_SLOTS`, is specialized for the GH200 platform and the batch-size
settings used in the paper experiments, especially `bsz=8` and `bsz=512`.

For other batch sizes or hardware platforms, the kernel configuration may need
to be retuned. In particular, update the corresponding parameters in both:

- `app/python/config.py`
- `include/task/config.cuh`

The parameters that commonly need adjustment include:

- `tile_m`, `tile_k`, and `tile_n`;
- `buffer_slots`;
- multicast-specific tile sizes and buffer slots;
- multicast `cluster_size`;
- the `MMA_Atom` configuration inside `gemv_1d` and `gemv_1d_multicast`.

After changing CUDA-side configuration values, rebuild the extension from the
repository root:

```bash
make pyext
```
