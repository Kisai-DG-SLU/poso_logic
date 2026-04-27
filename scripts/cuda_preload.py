"""
Preload CUDA driver before torch import
Fix for nvidia driver loading in containers with glibc mismatch
"""
import ctypes
import os

def preload_cuda():
    """Preload libcuda.so.1 before torch initialization"""
    try:
        cuda_paths = [
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/local/nvidia/lib64/libcuda.so.1",
        ]
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                ctypes.CDLL(cuda_path)
                break
    except Exception:
        pass

# Auto-preload on import
preload_cuda()