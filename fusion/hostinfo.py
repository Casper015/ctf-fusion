"""The host's CPU model, for benchmark reports (no third-party packages)."""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path


def cpu_name() -> str:
    """Marketing name of the CPU when the OS exposes it, else platform.processor()."""
    system = platform.system()
    try:
        if system == "Windows":
            import winreg  # noqa: PLC0415
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, ImportError, subprocess.SubprocessError, ValueError):
        pass
    return platform.processor() or platform.machine() or "unknown"


def profile() -> dict:
    return {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "os": platform.platform(), "machine": platform.machine()}
