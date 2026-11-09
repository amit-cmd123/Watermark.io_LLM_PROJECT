"""Install into the project venv despite Python mode-0700 temp ACL issues.

Only temporary directories under this project get inherited permissions.
No system ACLs, sandbox settings, or files outside the project are changed.
"""
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
temporary = root / ".runtime" / "tmp"
temporary.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
original_mkdir = os.mkdir


def mkdir(path, mode=0o777, *, dir_fd=None):
    absolute = Path(path).absolute()
    if mode == 0o700 and absolute.is_relative_to(temporary):
        mode = 0o777
    if dir_fd is None:
        return original_mkdir(path, mode)
    return original_mkdir(path, mode, dir_fd=dir_fd)


os.mkdir = mkdir
destination = root / ".venv" / "Lib" / "site-packages"
destination.mkdir(parents=True, exist_ok=True)
sys.argv = ["pip", "install", "--disable-pip-version-check", "--no-cache-dir",
            "--target", str(destination), *sys.argv[1:]]
runpy.run_module("pip", run_name="__main__")
