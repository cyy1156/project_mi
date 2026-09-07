"""方案 17 · 内存硬保护：后台线程监控，超限立即 os._exit。

阈值来自本机 Event 2004：python 曾提交 32GB+ 虚拟内存导致整机卡死。
默认阈值偏保守，适合 ~16GB 物理内存 / 可增长 pagefile（commit ~60G+）。
与 5060_three_hier_loss_accpaper/mem_guard.py 同实现。
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class MemGuardLimits:
    # 本进程虚拟地址 / 工作集上限（GB）
    max_process_virt_gb: float = 40.0
    max_process_ws_gb: float = 14.0
    # 系统可用物理内存下限；提交占用上限（相对 commit limit）
    min_sys_free_phys_gb: float = 0.12
    max_sys_commit_ratio: float = 0.98
    # 绝对提交占用上限（GB）；pagefile 可增长时用高天花板，靠 ratio + 进程上限防卡死
    max_sys_commit_used_gb: float = 60.0
    # 若 pagefile 尚未涨到配置上限，短暂顶满 commit_limit 时不立即杀（给 Windows 扩容时间）
    allow_pagefile_grow: bool = True
    poll_sec: float = 1.5


_STARTED = False
_STOP = threading.Event()


def _win_mem_gb() -> tuple[float, float, float, float]:
    """free_phys, free_pagefile, total_pagefile, load_pct."""
    import ctypes

    class MEMSTAT(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    m = MEMSTAT()
    m.dwLength = ctypes.sizeof(MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return (
        m.ullAvailPhys / (1024**3),
        m.ullAvailPageFile / (1024**3),
        m.ullTotalPageFile / (1024**3),
        float(m.dwMemoryLoad),
    )


def _process_mem_gb() -> tuple[float, float]:
    """(working_set_gb, commit_gb) for current PID."""
    try:
        import psutil

        mi = psutil.Process(os.getpid()).memory_info()
        # Windows: rss≈WorkingSet; private≈Private Bytes (~commit for most cases)
        ws = float(getattr(mi, "rss", 0)) / (1024**3)
        commit = float(getattr(mi, "private", getattr(mi, "vms", 0))) / (1024**3)
        return ws, commit
    except Exception:
        pass

    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    GetCurrentProcess = kernel32.GetCurrentProcess
    GetCurrentProcess.restype = wintypes.HANDLE
    GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
    GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    ok = GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        return 0.0, 0.0
    ws = counters.WorkingSetSize / (1024**3)
    commit = max(counters.PagefileUsage, counters.PrivateUsage) / (1024**3)
    return ws, commit


def _trip(reason: str) -> None:
    msg = f"[mem_guard] HARD STOP: {reason}"
    try:
        print(msg, flush=True)
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass
    # 不等 finally / CUDA 清理：尽快让外部看门狗与系统回血
    os._exit(99)


def _pagefile_can_grow() -> bool:
    """True if D:/C pagefile allocated size is still below configured max."""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management",
        ) as k:
            raw, _ = winreg.QueryValueEx(k, "PagingFiles")
        # entries like "d:\\pagefile.sys 4096 49152"
        cfg_max_mb = 0
        for line in raw if isinstance(raw, (list, tuple)) else [raw]:
            parts = str(line).split()
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                cfg_max_mb = max(cfg_max_mb, int(parts[2]))
        if cfg_max_mb <= 0:
            return False
        import subprocess

        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_PageFileUsage | Measure-Object AllocatedBaseSize -Sum).Sum",
            ],
            text=True,
            timeout=8,
        ).strip()
        allocated = float(out or "0")
        return allocated + 512 < float(cfg_max_mb)  # still room to grow
    except Exception:
        return False


def _loop(limits: MemGuardLimits) -> None:
    # 空闲内存允许短暂谷底（pack/CUDA 启动）；连续超限才熔断。
    # 进程 commit/WS 超限立即杀（防 Event 2004）。
    # pagefile 仍可增长时：不因短暂顶满当前 commit_limit 而误杀。
    free_strikes = 0
    free_strike_need = 8  # poll_sec≈1 → ~8s（折间谷底更宽松）
    free_strike_need_grow = 50  # ~50s：允许 pagefile 扩容 / 换页，避免一触即杀
    ratio_strikes = 0
    ratio_strike_need = 8  # ~8s；给 pagefile 扩容时间
    can_grow = bool(limits.allow_pagefile_grow)
    grow_checked = False
    while not _STOP.is_set():
        try:
            free_phys, free_pf, total_pf, load = _win_mem_gb()
            commit_used = max(0.0, total_pf - free_pf)
            ws, proc_commit = _process_mem_gb()
            if not grow_checked and limits.allow_pagefile_grow:
                can_grow = _pagefile_can_grow()
                grow_checked = True
                print(
                    f"[mem_guard] pagefile_can_grow={can_grow}",
                    flush=True,
                )

            if proc_commit >= limits.max_process_virt_gb:
                _trip(
                    f"process_commit={proc_commit:.2f}G >= {limits.max_process_virt_gb}G "
                    f"(ws={ws:.2f}G sys_free={free_phys:.2f}G)"
                )
            if ws >= limits.max_process_ws_gb:
                _trip(
                    f"process_ws={ws:.2f}G >= {limits.max_process_ws_gb}G "
                    f"(commit={proc_commit:.2f}G sys_free={free_phys:.2f}G)"
                )
            if free_phys < limits.min_sys_free_phys_gb:
                free_strikes += 1
                need_free = free_strike_need_grow if can_grow else free_strike_need
                if can_grow and free_strikes % 15 == 0:
                    can_grow = _pagefile_can_grow()
                    need_free = free_strike_need_grow if can_grow else free_strike_need
                if free_strikes >= need_free:
                    _trip(
                        f"sys_free_phys={free_phys:.2f}G < {limits.min_sys_free_phys_gb}G "
                        f"for {free_strikes} polls "
                        f"(proc_commit={proc_commit:.2f}G load={load:.0f}% grow={int(can_grow)})"
                    )
            else:
                free_strikes = 0
            if commit_used >= limits.max_sys_commit_used_gb:
                _trip(
                    f"sys_commit_used={commit_used:.2f}G >= {limits.max_sys_commit_used_gb}G "
                    f"(limit={total_pf:.2f}G proc={proc_commit:.2f}G)"
                )
            ratio_hot = total_pf > 0 and (commit_used / total_pf) >= limits.max_sys_commit_ratio
            if ratio_hot:
                ratio_strikes += 1
                # 可增长：多等几次，并每 ~8s 重查 allocated 是否已涨
                need = ratio_strike_need if can_grow else 2
                if can_grow and ratio_strikes % ratio_strike_need == 0:
                    can_grow = _pagefile_can_grow()
                if (not can_grow) and ratio_strikes >= need:
                    _trip(
                        f"sys_commit_ratio={commit_used/total_pf:.1%} >= "
                        f"{limits.max_sys_commit_ratio:.0%} "
                        f"(used={commit_used:.2f}/{total_pf:.2f}G grow=0)"
                    )
            else:
                ratio_strikes = 0
        except Exception as exc:
            try:
                print(f"[mem_guard] poll error: {exc}", flush=True)
            except Exception:
                pass
        _STOP.wait(limits.poll_sec)


def start_mem_guard(limits: MemGuardLimits | None = None) -> None:
    """Idempotent：进程内只启一次守护线程。"""
    global _STARTED
    if _STARTED:
        return
    lim = limits or MemGuardLimits()
    free_phys, free_pf, total_pf, load = _win_mem_gb()
    commit_used = max(0.0, total_pf - free_pf)
    print(
        f"[mem_guard] ON max_proc_commit={lim.max_process_virt_gb}G "
        f"max_proc_ws={lim.max_process_ws_gb}G "
        f"min_sys_free={lim.min_sys_free_phys_gb}G "
        f"max_sys_commit={lim.max_sys_commit_used_gb}G "
        f"| now free={free_phys:.2f}G commit={commit_used:.2f}/{total_pf:.2f}G "
        f"load={load:.0f}%",
        flush=True,
    )
    # 启动前也做一次硬门闸（与 run_arm 预检互补）
    if free_phys < 3.0:
        raise SystemExit(
            f"[mem_guard] refuse start: avail_phys={free_phys:.2f}G < 3.0G"
        )
    if (not lim.allow_pagefile_grow) and commit_used > lim.max_sys_commit_used_gb - 2.0:
        raise SystemExit(
            f"[mem_guard] refuse start: commit_used={commit_used:.2f}G "
            f"too close to cap {lim.max_sys_commit_used_gb}G"
        )
    t = threading.Thread(target=_loop, args=(lim,), name="mem_guard", daemon=True)
    t.start()
    _STARTED = True


def stop_mem_guard() -> None:
    _STOP.set()
