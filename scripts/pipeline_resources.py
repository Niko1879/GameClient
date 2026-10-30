"""Read memory of this benchmark's process descendants using Windows APIs."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'scripts'))
from event_resources import snapshot

class Entry(ctypes.Structure):
    _fields_=[('size',wintypes.DWORD),('usage',wintypes.DWORD),('pid',wintypes.DWORD),
        ('heap',ctypes.c_size_t),('module',wintypes.DWORD),('threads',wintypes.DWORD),
        ('parent',wintypes.DWORD),('priority',wintypes.LONG),('flags',wintypes.DWORD),('exe',wintypes.WCHAR*260)]

class Memory(ctypes.Structure):
    _fields_=[('size',wintypes.DWORD),('faults',wintypes.DWORD)]+[(name,ctypes.c_size_t)
        for name in ('peak_working','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]

kernel,psapi=ctypes.windll.kernel32,ctypes.windll.psapi
kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
kernel.OpenProcess.restype=wintypes.HANDLE
kernel.CloseHandle.argtypes=[wintypes.HANDLE]
kernel.Process32FirstW.argtypes=kernel.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)]
psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]

def tree():
    handle=kernel.CreateToolhelp32Snapshot(2,0)
    assert handle!=ctypes.c_void_p(-1).value
    entry=Entry();entry.size=ctypes.sizeof(entry)
    parents={}
    try:
        found=kernel.Process32FirstW(handle,ctypes.byref(entry))
        while found:
            parents[entry.pid]=entry.parent
            found=kernel.Process32NextW(handle,ctypes.byref(entry))
    finally:kernel.CloseHandle(handle)
    owned={os.getpid()}
    while True:
        following=owned|{pid for pid,parent in parents.items() if parent in owned}
        if following==owned:break
        owned=following
    private=working=0
    for pid in owned:
        process=kernel.OpenProcess(0x410,False,pid)
        if not process:continue
        try:
            memory=Memory();memory.size=ctypes.sizeof(memory)
            if psapi.GetProcessMemoryInfo(process,ctypes.byref(memory),memory.size):
                private+=memory.private;working+=memory.working
        finally:kernel.CloseHandle(process)
    return dict(**snapshot(HERE),aggregate_private_bytes=private,aggregate_working_bytes=working,process_count=len(owned))
