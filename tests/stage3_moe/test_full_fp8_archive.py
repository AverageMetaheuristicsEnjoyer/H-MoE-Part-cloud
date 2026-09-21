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
    assert uploads[0]["folder_path"] == str(checkpoint.parents[1])
    assert checkpoint.read_bytes() == payload
