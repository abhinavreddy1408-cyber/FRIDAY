"""
F.R.I.D.A.Y. Windows Startup Installer
======================================
Registers the assistant to start automatically at logon using Windows Task Scheduler.
"""

import ctypes
import os
import subprocess
import sys


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False


def install_startup():
    if not is_admin():
        print("!!! ERROR: Please run this script as Administrator to register startup task.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    watchdog_path = os.path.join(script_dir, "friday", "watchdog.py")
    pythonw_path = sys.executable.replace("python.exe", "pythonw.exe")

    task_name = "FRIDAY_Assistant"
    # Command to run: pythonw.exe watchdog.py
    # schtasks /create /tn "FRIDAY_Assistant" /tr "pythonw_path watchdog_path" /sc onlogon /rl highest /f

    cmd = [
        "schtasks", "/create",
        "/tn", task_name,
        "/tr", f'"{pythonw_path}" "{watchdog_path}"',
        "/sc", "onlogon",
        "/rl", "highest",
        "/f"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("--- SUCCESS ---")
        print("F.R.I.D.A.Y. has been registered to start at logon.")
        print(f"Task Name: {task_name}")
        print(f"Command: {pythonw_path} {watchdog_path}")
    except subprocess.CalledProcessError as e:
        print("--- FAILURE ---")
        print(f"Could not register startup task: {e}")

if __name__ == "__main__":
    install_startup()
