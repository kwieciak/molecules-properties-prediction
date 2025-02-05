import psutil
import torch


def print_memory_usage():
    ram_memory= psutil.virtual_memory()
    free_ram = ram_memory.available / (1024 ** 3)
    total_ram = ram_memory.total / (1024 ** 3)
    print(f"Free RAM: {free_ram:.2f} GB / {total_ram:.2f} GB")

def print_gpu_memory():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"VRAM usage: {gpu_memory:.2f} GB")