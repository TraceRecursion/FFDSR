import torch
import os

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA (NVIDIA GPU)"
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (Apple GPU)"
    return torch.device("cpu"), "CPU"