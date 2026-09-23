"""Basic installation test: checks that the DAK packages import and see a GPU.

Usage:
    python test_install.py
"""

import os
import sys

# Drop the repo root from sys.path so the installed packages are tested,
# not the source directories with the same names.
_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _repo_root]

import torch

print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
assert torch.cuda.is_available(), "CUDA is not available"
print(f"GPU: {torch.cuda.get_device_name(0)}")

import offload
from offload import runtime

print(f"offload: OK ({offload.__file__})")

import opt_attention
from opt_attention import _C

print(f"opt_attention: OK ({opt_attention.__file__})")

print("Installation test PASSED")
