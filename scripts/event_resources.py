"""Small local resource guard for bounded event preparation and training."""
from __future__ import annotations

import ctypes
import os
import shutil


def snapshot(path, model=None):
    result = {'free_disk_bytes': shutil.disk_usage(path).free}
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total',
                'page_available', 'virtual_total', 'virtual_available', 'extended')]
        memory = Memory(); memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            result.update(total_ram_bytes=memory.total, available_ram_bytes=memory.available)
        class Counters(ctypes.Structure):
            _fields_ = [('size', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ('peak_working', 'working', 'peak_paged',
                'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]
        counters = Counters(); counters.size = ctypes.sizeof(counters)
        function = ctypes.windll.psapi.GetProcessMemoryInfo
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if function(ctypes.c_void_p(-1), ctypes.byref(counters), counters.size):
            result['private_bytes'] = counters.private
    if model is not None and model.device.type == 'cuda':
        import torch
        result.update(gpu_allocated_bytes=torch.cuda.memory_allocated(model.device),
                      gpu_peak_bytes=torch.cuda.max_memory_allocated(model.device))
    return result


def within_budget(resource, *, minimum_ram=4*1024**3, maximum_private=8*1024**3,
                  minimum_disk=80*1024**3, maximum_gpu=9*1024**3):
    # Defaults suit the 16 GB / RX 6750 XT workstation; larger machines pass their own.
    return (resource['free_disk_bytes'] >= minimum_disk
            and resource.get('available_ram_bytes', minimum_ram) >= minimum_ram
            and resource.get('private_bytes', 0) <= maximum_private
            and resource.get('gpu_allocated_bytes', 0) <= maximum_gpu)
