#!/usr/bin/env bash
set -euo pipefail

work_root=${NODE207_MONARCH_ROOT:-/var/tmp/user1-monarch-pretrain}
python_bin=${NODE207_MONARCH_PYTHON:-$work_root/venv-py312/bin/python}
loader=/lib64/ld-linux-x86-64.so.2

nvidia_root=$("$python_bin" -c 'import site; from pathlib import Path; print(next(Path(path) / "nvidia" for path in site.getsitepackages() if (Path(path) / "nvidia").is_dir()))')
nvidia_lib_path=$(find "$nvidia_root" -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
site_packages=${nvidia_root%/nvidia}

exec "$loader" --inhibit-cache \
  --library-path "$nvidia_lib_path:$site_packages/torch/lib" \
  "$python_bin" "$@"
