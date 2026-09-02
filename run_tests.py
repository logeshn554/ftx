import os
import pathlib
import sys

orig_stat = os.stat
def safe_os_stat(path, *args, **kwargs):
    try:
        return orig_stat(path, *args, **kwargs)
    except PermissionError:
        return os.stat_result((0o040777, 0, 0, 0, 0, 0, 0, 0, 0, 0))

os.stat = safe_os_stat

class EmptyScandir:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __iter__(self):
        return self
    def __next__(self):
        raise StopIteration

orig_scandir = os.scandir
def safe_scandir(path=None):
    try:
        return orig_scandir(path)
    except PermissionError:
        return EmptyScandir()

os.scandir = safe_scandir

orig_listdir = os.listdir
def safe_listdir(path=None):
    try:
        return orig_listdir(path)
    except PermissionError:
        return []

os.listdir = safe_listdir

orig_rmdir = os.rmdir
def safe_rmdir(path, *args, **kwargs):
    try:
        return orig_rmdir(path, *args, **kwargs)
    except (PermissionError, FileNotFoundError):
        pass

os.rmdir = safe_rmdir

_dir_counter = 0
def custom_make_numbered_dir(root, prefix, mode=0o700):
    global _dir_counter
    _dir_counter += 1
    p = pathlib.Path(root) / f"{prefix}_{_dir_counter}"
    os.makedirs(str(p), exist_ok=True)
    return p

try:
    import _pytest.pathlib
    _pytest.pathlib.make_numbered_dir = custom_make_numbered_dir
    _pytest.pathlib._force_symlink = lambda *args, **kwargs: None
except Exception:
    pass

import pytest
if __name__ == "__main__":
    local_temp = os.path.abspath(os.path.join(os.path.dirname(__file__), ".pytest_tmp"))
    custom_args = ["tests", "-p", "no:cacheprovider", f"--basetemp={local_temp}", "-q"]
    args = sys.argv[1:] if len(sys.argv) > 1 else custom_args
    sys.exit(pytest.main(args))
