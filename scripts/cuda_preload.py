"""
Preload CUDA driver before torch import
Fix for nvidia driver loading in containers with glibc mismatch
"""
import ctypes
import os

def preload_cuda():
    """Preload libcuda.so.1 before torch initialization"""
    try:
        cuda_path = "/usr/local/nvidia/lib64/libcuda.so.1"
        if os.path.exists(cuda_path):
            ctypes.CDLL(cuda_path)
    except Exception:
        pass

# Auto-preload on import
preload_cuda()