import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage3_moe.slim_checkpoint_archive import archive, restore


@pytest.mark.parametrize('corruption', [None, 'size', 'sha256', 'missing'])
def test_archive_verifies_remote_before_publishing_resume(tmp_path, corruption):
    checkpoint = tmp_path / 'checkpoint'
    path = checkpoint / 'iter_0000322/mp_rank_00/model_optim_rng.pt'
    path.parent.mkdir(parents=True)
    payload = b'full weights optimizer rng'
    path.write_bytes(payload)
    entry = SimpleNamespace(path='test/iter_0000322/mp_rank_00/model_optim_rng.pt',
        size=len(payload) + (corruption == 'size'),
        lfs=SimpleNamespace(sha256='bad' if corruption == 'sha256' else hashlib.sha256(payload).hexdigest()))
    pointers = []
    api = SimpleNamespace(upload_folder=lambda **kw: None,
        list_repo_tree=lambda *a, **kw: [] if corruption == 'missing' else [entry],
        upload_file=lambda **kw: pointers.append(json.loads(kw['path_or_fileobj'])))
    if corruption:
        with pytest.raises(RuntimeError, match='verification failed'):
            archive(checkpoint, 322, 'test', api)
        assert pointers == []
    else:
        archive(checkpoint, 322, 'test', api)
        assert pointers[0]['iteration'] == 322
    assert path.read_bytes() == payload


@pytest.mark.parametrize('corrupt', [False, True])
def test_restore_verifies_before_publishing_tracker(tmp_path, monkeypatch, corrupt):
    payload = b'weights optimizer rng'
    blob = tmp_path / 'blob.pt'
    blob.write_bytes(b'bad' if corrupt else payload)
    pointer = tmp_path / 'resume.json'
    pointer.write_text(json.dumps({'iteration': 322, 'files': {
        'mp_rank_00/model_optim_rng.pt': [len(payload), hashlib.sha256(payload).hexdigest()]}}))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        hf_hub_download=lambda repo, filename, **kw: str(pointer if filename.endswith('resume.json') else blob)))
    api = SimpleNamespace(file_exists=lambda *a, **kw: True, token='test')
    checkpoint = tmp_path / 'checkpoint'
    if corrupt:
        with pytest.raises(RuntimeError, match='verification'):
            restore(checkpoint, tmp_path / 'logs', 'test', api)
        assert not (checkpoint / 'latest_checkpointed_iteration.txt').exists()
    else:
        restore(checkpoint, tmp_path / 'logs', 'test', api)
        assert (checkpoint / 'latest_checkpointed_iteration.txt').read_text() == '322'
        assert (checkpoint / 'iter_0000322/mp_rank_00/model_optim_rng.pt').read_bytes() == payload


def test_missing_archive_does_not_silently_restart_existing_training(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=None))
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'train-old.log').write_text('iteration 300')
    with pytest.raises(RuntimeError, match='refusing fresh restart'):
        restore(tmp_path / 'checkpoint', logs, 'test', SimpleNamespace(file_exists=lambda *a, **kw: False))


def test_full_training_restores_seed_and_requires_full_archive(tmp_path, monkeypatch):
    from stage3_moe import slim_checkpoint_archive as module
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    tracker = checkpoint / 'latest_checkpointed_iteration.txt'
    calls = []
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=lambda **kw: object()))

    def load(checkpoint, run_dir, prefix, api):
        calls.append(('restore', prefix))
        if prefix == 'short/split':
            tracker.write_text('2254')
            return {'iteration': 2254}

    def launch(*args, **kwargs):
        assert tracker.read_text() == '2254'
        tracker.write_text('17242')
        return SimpleNamespace(poll=lambda: 0)

    def save(checkpoint, step, prefix, api):
        calls.append(('archive', step, prefix))
        return {'iteration': step}

    monkeypatch.setattr(module, 'restore', load)
    monkeypatch.setattr(module, 'archive', save)
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    module.train([], {'HF_TOKEN': 'test'}, tmp_path, checkpoint, tmp_path / 'logs',
                 'full/split', target=17242, seed_prefix='short/split')
    assert calls == [('restore', 'full/split'), ('restore', 'short/split'),
                     ('archive', 17242, 'full/split')]
