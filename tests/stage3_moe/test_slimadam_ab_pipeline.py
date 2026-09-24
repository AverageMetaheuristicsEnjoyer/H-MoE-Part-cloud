import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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
