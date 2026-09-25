import runpy
import subprocess
import sys
import importlib.util
import shutil
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_te4_import_preserves_image_paths_and_discovers_nvidia_libraries(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIM_AB_ROOT", str(tmp_path / "output"))
    monkeypatch.setenv("MLSUB_IMAGE", "te4")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/image/cuda/lib64")
    for variable in ("CUDNN_HOME", "CURAND_HOME", "NVRTC_HOME"):
        monkeypatch.delenv(variable, raising=False)
    nvidia = tmp_path / "site-packages" / "nvidia"
    for package in ("cuda_runtime", "cudnn", "curand", "cuda_nvrtc"):
        (nvidia / package / "lib").mkdir(parents=True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(nvidia)]))
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 1024**3))
    original_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: str(path).startswith("/home/jovyan/data/") or original_is_file(path))
    monkeypatch.setattr(sys, "argv", ["cloud_slimadam_ab.py", "smoke", "split"])
    captured = []

    def child(command, **kwargs):
        if "-c" in command:
            captured.append(kwargs["env"])
            raise RuntimeError("Stop after capturing the runtime import environment")

    monkeypatch.setattr(subprocess, "run", child)
    runpy.run_path(str(ROOT / "scripts/cloud_slimadam_ab.py"), run_name="__main__")
    assert len(captured) == 1
    env = captured[0]
    assert "/image/cuda/lib64" in env["LD_LIBRARY_PATH"].split(":")
    assert str(nvidia / "cuda_runtime" / "lib") in env["LD_LIBRARY_PATH"].split(":")
    assert env["CUDNN_HOME"] == str(nvidia / "cudnn")
    assert env["CURAND_HOME"] == str(nvidia / "curand")
    assert env["NVRTC_HOME"] == str(nvidia / "cuda_nvrtc")


def test_pipeline_does_not_train_after_smoke_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SLIM_AB_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["cloud_slimadam_ab.py", "pipeline", "split"])
    stages = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: stages.append(command[2]))
    runpy.run_path(str(ROOT / "scripts/cloud_slimadam_ab.py"), run_name="__main__")
    assert stages == ["smoke"]
    assert "EXIT=1" in capsys.readouterr().out


def test_pipeline_requires_both_success_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SLIM_AB_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["cloud_slimadam_ab.py", "pipeline", "split"])
    stages = []

    def child(command, **kwargs):
        stage = command[2]
        stages.append(stage)
        experiment = "slimadam-fc1-ab-20260924-v1"
        directory = tmp_path / experiment / "logs" / (
            f"stage3-slimadam_bf16_state_fp32-slim-ab-{experiment}-{stage}-split"
        )
        directory.mkdir(parents=True)
        artifact = "smoke-pass.json" if stage == "smoke" else "endpoint.json"
        (directory / artifact).write_text("{}")

    monkeypatch.setattr(subprocess, "run", child)
    runpy.run_path(str(ROOT / "scripts/cloud_slimadam_ab.py"), run_name="__main__")
    assert stages == ["smoke", "train"]
    assert "PIPELINE_RESULT=COMPLETE" in capsys.readouterr().out
