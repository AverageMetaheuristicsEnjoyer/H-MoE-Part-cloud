import os

import torch
import torch.distributed as dist


def main() -> None:
    if not __debug__:
        raise RuntimeError("run smoke tests without Python optimization")

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    assert world_size == 2, f"expected WORLD_SIZE=2, got {world_size}"
    assert rank in (0, 1), rank
    assert local_rank in (0, 1), local_rank
    assert torch.cuda.is_available()

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    assert capability == (9, 0), capability

    dist.init_process_group(backend="nccl")
    try:
        assert dist.get_backend() == "nccl", dist.get_backend()
        assert dist.get_world_size() == 2, dist.get_world_size()
        assert dist.get_rank() == rank, (dist.get_rank(), rank)

        value = torch.eye(16, device=device, dtype=torch.float32).mul_(rank + 1)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        expected = torch.eye(16, device=device, dtype=torch.float32).mul_(3)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)

        print(
            f"nccl_identity=pass rank={rank} local_rank={local_rank}"
            f" device={torch.cuda.get_device_name(device)} sm={capability[0]}{capability[1]}",
            flush=True,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
