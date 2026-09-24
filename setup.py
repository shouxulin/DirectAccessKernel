from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import shutil

import torch
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

extra_link_args = [
    f"-Wl,-rpath,{torch_lib}",
    # f"-L{torch_lib}",
    # "-ltorch_cuda",
    # "-lc10_cuda",
    # "-ltorch_cpu",
    # "-lc10",
]

this_dir = os.path.dirname(os.path.abspath(__file__))
sources = [os.path.join(this_dir, "src", "torch_runtime.cu")]
runtime_obj = os.path.join(this_dir, "runtime.o")
include_dirs = [
    os.path.join(this_dir, "include"),
]

# CUDA architecture, passed in by `make pyext arch=<arch>` (see Makefile); must match runtime.o
arch = os.environ.get("DAK_ARCH", "90a")
supported_archs = ("90a", "120a")
if arch not in supported_archs:
    raise ValueError(f"Unsupported DAK_ARCH '{arch}', expected one of: {', '.join(supported_archs)}")
arch_flags = [
    f"-gencode=arch=compute_{arch},code=sm_{arch}",
    f"-DDAK_SM_ARCH={arch.rstrip('a')}",
]

setup(
    name="llm-offload",

    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=[
        CUDAExtension(
            name = "offload.runtime",

            sources=sources,
            extra_objects=[runtime_obj],
            depends=[runtime_obj],  # rebuild when runtime.o is rebuilt, e.g. for another arch
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DNDEBUG"],
                "nvcc": [
                    *arch_flags,
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                    "-Xptxas=-v"
                ],
            },
            libraries=["cuda"],
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
