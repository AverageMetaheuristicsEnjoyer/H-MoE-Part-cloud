import argparse
import json
import time
from pathlib import Path

import torch


DATA_KEYS = {"exp_avg", "exp_avg_sq", "momentum_buffer"}


def unique_bytes(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    return sum(storages.values())


def optimizer_blocks(optimizer):
    return optimizer if isinstance(optimizer, list) else [optimizer]


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
args = parser.parse_args()

start = time.perf_counter()
checkpoint = torch.load(
    args.checkpoint,
    map_location="cpu",
    weights_only=False,
    mmap=True,
)
load_seconds = time.perf_counter() - start

blocks = []
all_state_data = []
all_metadata = []
all_masters = []
for index, block in enumerate(optimizer_blocks(checkpoint["optimizer"])):
    state_data = []
    metadata = []
    state_dtypes = set()
    metadata_dtypes = set()
    for state in block["optimizer"]["state"].values():
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            if key in DATA_KEYS:
                state_data.append(value)
                state_dtypes.add(f"{key}:{value.dtype}")
            else:
                metadata.append(value)
                metadata_dtypes.add(f"{key}:{value.dtype}")
    masters = [
        tensor
        for group in block["fp32_from_fp16_params"]
        for tensor in group
    ]
    all_state_data.extend(state_data)
    all_metadata.extend(metadata)
    all_masters.extend(masters)
    blocks.append(
        {
            "index": index,
            "parameter_states": len(block["optimizer"]["state"]),
            "state_data_bytes": unique_bytes(state_data),
            "metadata_bytes": unique_bytes(metadata),
            "master_parameter_bytes": unique_bytes(masters),
            "state_dtypes": sorted(state_dtypes),
            "metadata_dtypes": sorted(metadata_dtypes),
        }
    )

model_tensors = [value for value in checkpoint["model"].values() if torch.is_tensor(value)]
result = {
    "checkpoint": str(args.checkpoint),
    "checkpoint_file_bytes": args.checkpoint.stat().st_size,
    "mmap_deserialize_seconds": load_seconds,
    "iteration": checkpoint["iteration"],
    "model_bytes": unique_bytes(model_tensors),
    "optimizer_state_data_bytes": unique_bytes(all_state_data),
    "optimizer_metadata_bytes": unique_bytes(all_metadata),
    "optimizer_state_total_bytes": unique_bytes(all_state_data + all_metadata),
    "master_parameter_bytes": unique_bytes(all_masters),
    "blocks": blocks,
}
print(json.dumps(result, indent=2))
