import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("corruption", [None, "size", "sha256", "missing"])
def test_checkpoint_archive_requires_matching_remote_hash_and_size(tmp_path, monkeypatch, corruption):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=object))
    monkeypatch.setattr(sys, "argv", ["cloud_moe_full_fp8.py", "muon_bf16_state_fp8"])
    script = Path(__file__).resolve().parents[2] / "scripts/cloud_moe_full_fp8.py"
    spec = importlib.util.spec_from_file_location("full_fp8_archive", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CHECKPOINT = tmp_path
    checkpoint = tmp_path / "iter_0013794/mp_rank_00/model_optim_rng.pt"
    checkpoint.parent.mkdir(parents=True)
    payload = b"complete model, optimizer, and rng checkpoint fixture"
    checkpoint.write_bytes(payload)
    remote = f"{module.PREFIX}/iter_0013794/mp_rank_00/model_optim_rng.pt"
    uploads = []
    entry = SimpleNamespace(
        path=remote,
        size=len(payload) + (corruption == "size"),
        lfs=SimpleNamespace(sha256="bad" if corruption == "sha256" else hashlib.sha256(payload).hexdigest()),
    )
    api = SimpleNamespace(
        upload_folder=lambda **kwargs: uploads.append(kwargs),
        list_repo_tree=lambda *args, **kwargs: [] if corruption == "missing" else [entry],
    )
    if corruption:
        with pytest.raises(RuntimeError, match="verification failed"):
            module.archive(13794, api)
    else:
        module.archive(13794, api)
    assert uploads[0]["folder_path"] == str(tmp_path.parent / ".muon_bf16_state_fp8-archive-13794")
    assert checkpoint.read_bytes() == payload

@pytest.fixture
def recovery_module(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=object))
    monkeypatch.setattr(sys, "argv", ["cloud_moe_full_fp8.py", "muon_bf16_state_fp8"])
    script = Path(__file__).resolve().parents[2] / "scripts/cloud_moe_full_fp8.py"
    spec = importlib.util.spec_from_file_location("full_fp8_recovery", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CHECKPOINT = tmp_path / "checkpoints"
    module.LOG_ROOT = tmp_path / "logs"
    module.env["HF_TOKEN"] = "test-token"
    return module


def test_missing_checkpoint_after_prior_full_run_refuses_fresh_start(recovery_module):
    m = recovery_module
    logs = m.LOG_ROOT / f"stage3-{m.ARM}-full-{m.WAVE}"
    logs.mkdir(parents=True)
    (logs / "train-before-restart.log").write_text("iteration 7000")
    with pytest.raises(RuntimeError, match="refusing fresh restart"):
        m.restore(SimpleNamespace(file_exists=lambda *a, **kw: False))


@pytest.mark.parametrize("corrupt", [False, True])
def test_restore_remote_checkpoint_verifies_before_publishing_tracker(recovery_module, tmp_path, monkeypatch, corrupt):
    import json
    m = recovery_module
    payload = b"weights optimizer rng"
    blob = tmp_path / "blob.pt"
    blob.write_bytes(payload if not corrupt else b"broken")
    pointer = tmp_path / "resume.json"
    pointer.write_text(json.dumps({"iteration": 363, "files": {
        "mp_rank_00/model_optim_rng.pt": [len(payload), hashlib.sha256(payload).hexdigest()]
    }}))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        hf_hub_download=lambda repo, filename, **kw: str(pointer if filename.endswith("resume.json") else blob)
    ))
    api = SimpleNamespace(file_exists=lambda *a, **kw: True)
    if corrupt:
        with pytest.raises(RuntimeError, match="restore verification failed"):
            m.restore(api)
        assert not (m.CHECKPOINT / "latest_checkpointed_iteration.txt").exists()
    else:
        m.restore(api)
        assert (m.CHECKPOINT / "latest_checkpointed_iteration.txt").read_text() == "363"
        assert (m.CHECKPOINT / "iter_0000363/mp_rank_00/model_optim_rng.pt").read_bytes() == payload


def test_new_wave_without_prior_training_may_start_fresh(recovery_module):
    assert recovery_module.restore(SimpleNamespace(file_exists=lambda *a, **kw: False)) is None


def test_resume_pointer_preserves_verified_predecay(recovery_module):
    import json
    uploaded = []
    api = SimpleNamespace(upload_file=lambda **kw: uploaded.append(json.loads(kw["path_or_fileobj"])))
    m = recovery_module
    first = m.save_resume_pointer(13794, {"x.pt": [10, "hash"]}, api)
    second = m.save_resume_pointer(14157, {"x.pt": [10, "hash"]}, api, first)
    assert second["verified_retained"] == [13794]
    assert uploaded[-1]["iteration"] == 14157


def test_live_hf_archive_restore_roundtrip(tmp_path, monkeypatch):
    import os
    import uuid
    import shutil
    if os.environ.get("STAGE3_FP8_LIVE_ARCHIVE_TEST") != "1":
        pytest.skip("requires explicitly enabled live HF roundtrip")
    from huggingface_hub import HfApi
    monkeypatch.setattr(sys, "argv", ["cloud_moe_full_fp8.py", "muon_bf16_state_fp8"])
    script = Path(__file__).resolve().parents[2] / "scripts/cloud_moe_full_fp8.py"
    spec = importlib.util.spec_from_file_location("live_fp8_archive", script)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.CHECKPOINT = tmp_path / "checkpoint"
    m.LOG_ROOT = tmp_path / "logs"
    m.PREFIX = "fp8-recovery-selftest/" + uuid.uuid4().hex
    target = m.CHECKPOINT / "iter_0000363/mp_rank_00/model_optim_rng.pt"
    target.parent.mkdir(parents=True)
    payload = os.urandom(4096)
    target.write_bytes(payload)
    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        files = m.archive(363, api)
        m.save_resume_pointer(363, files, api)
        shutil.rmtree(m.CHECKPOINT)
        m.restore(api)
        assert target.read_bytes() == payload
        assert (m.CHECKPOINT / "latest_checkpointed_iteration.txt").read_text() == "363"
    finally:
        api.delete_folder(repo_id=m.REPO, path_in_repo=m.PREFIX, repo_type="model")
