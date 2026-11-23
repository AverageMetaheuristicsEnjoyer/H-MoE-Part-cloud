#!/bin/bash
echo "RANK=${OMPI_COMM_WORLD_RANK:-na} SIZE=${OMPI_COMM_WORLD_SIZE:-na} LOCAL=${OMPI_COMM_WORLD_LOCAL_RANK:-na}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
python3 -c "import torch;print('torch_devcount', torch.cuda.device_count())" 2>&1 | tail -2
echo rankdone
