import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[2]
MPI_NATIVE = ROOT / "scripts" / "cloud_nccl_smoke_mpi_native.sh"
RANK0_TORCHRUN = ROOT / "scripts" / "cloud_nccl_smoke_rank0_torchrun.sh"


def fake_commands(tmp_path):
    python = tmp_path / "python"
    python.write_text(
        "#!/bin/sh\n"
        "echo PYTHON_ARGS=\"$*\"\n"
        "echo DIST_ENV=$RANK,$LOCAL_RANK,$WORLD_SIZE,$MASTER_ADDR,$MASTER_PORT\n"
    )
    python.chmod(0o755)
    nvidia_smi = tmp_path / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nprintf '0\\n1\\n'\n")
    nvidia_smi.chmod(0o755)
    return {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}


def test_mpi_native_maps_cloud_rank_environment(tmp_path):
    for world_size in (2, 4, 8):
        rank = world_size - 1
        env = fake_commands(tmp_path)
        env.update(
            OMPI_COMM_WORLD_RANK=str(rank),
            OMPI_COMM_WORLD_LOCAL_RANK=str(rank),
            OMPI_COMM_WORLD_SIZE=str(world_size),
        )

        completed = subprocess.run(
            [MPI_NATIVE], cwd=ROOT, env=env, check=True, text=True, capture_output=True
        )

        assert (
            f"LAUNCH=mpi-native rank={rank} local_rank={rank} world_size={world_size}"
            in completed.stdout
        )
        assert "PYTHON_ARGS=scripts/nccl_smoke.py" in completed.stdout
        assert f"DIST_ENV={rank},{rank},{world_size},127.0.0.1,29500" in completed.stdout


def test_rank0_torchrun_starts_one_worker_group(tmp_path):
    env = fake_commands(tmp_path)
    env.update(OMPI_COMM_WORLD_RANK="0", OMPI_COMM_WORLD_SIZE="2")

    completed = subprocess.run(
        [RANK0_TORCHRUN], cwd=ROOT, env=env, check=True, text=True, capture_output=True
    )

    assert "LAUNCH=rank0-torchrun outer_rank=0 workers=2" in completed.stdout
    assert "PYTHON_ARGS=-m torch.distributed.run --standalone --nproc-per-node 2" in completed.stdout


def test_rank0_torchrun_leaves_the_other_outer_rank_idle(tmp_path):
    env = fake_commands(tmp_path)
    env.update(OMPI_COMM_WORLD_RANK="1", OMPI_COMM_WORLD_SIZE="2")

    completed = subprocess.run(
        [RANK0_TORCHRUN], cwd=ROOT, env=env, check=True, text=True, capture_output=True
    )

    assert completed.stdout.strip() == "LAUNCH=rank0-torchrun outer_rank=1 action=idle"
    assert "PYTHON_ARGS" not in completed.stdout
