#!/bin/bash
R=/workspace-SR006.nfs3/hmoe-lionmoe-20261123/fullrun-016-v041/arms/lm3block
L=$(ls -t $R/logs/*/train-*.log 2>/dev/null | head -1); echo "LOG=$L"
echo "== traceback around last_print_t"
grep -a -B6 -A6 "last_print_t" "$L" 2>/dev/null | head -40
echo "== tqdm import (image)"
python -c "import tqdm,sys; print(tqdm.__version__, tqdm.__file__)" 2>&1 | tail -2
echo "== tqdm with user base"
PYTHONUSERBASE=/home/jovyan/.local-te4 python -c "import tqdm; print(tqdm.__version__, tqdm.__file__)" 2>&1 | tail -2
echo "== user base tqdm dirs"
ls -d /home/jovyan/.local-te4/lib/python3.12/site-packages/tqdm* 2>&1 | head -5
echo "== done"
