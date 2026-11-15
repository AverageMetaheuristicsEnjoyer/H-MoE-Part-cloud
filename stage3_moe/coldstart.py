import hashlib
import json
import os
from pathlib import Path

import torch


def install_initialization_receipt():
    import megatron.training.training as training
    from megatron.training import get_args

    original = training.setup_model_and_optimizer

    def setup(*args, **kwargs):
        models, optimizer, scheduler = original(*args, **kwargs)
        config = get_args()
        assert config.load is None and config.iteration == 0
        assert config.consumed_train_samples == 0
        assert config.seed == int(os.environ['STAGE3_MOE_SEED'])
        digest = hashlib.sha256()
        count = 0
        biases = {}
        for chunk, model in enumerate(models):
            for name, parameter in model.named_parameters():
                tensor = parameter.detach().cpu().contiguous()
                digest.update(json.dumps([chunk, name, list(tensor.shape), str(tensor.dtype)]).encode())
                digest.update(tensor.view(torch.uint8).numpy().tobytes())
                count += parameter.numel()
            for name, buffer in model.named_buffers():
                if name.endswith('expert_bias'):
                    biases[f'{chunk}:{name}'] = float(buffer.abs().max())
        assert len(biases) == 17 and all(value == 0 for value in biases.values())
        row = {
            'seed': config.seed, 'iteration': config.iteration,
            'consumed_train_samples': config.consumed_train_samples, 'load': config.load,
            'parameter_sha256': digest.hexdigest(), 'parameters': count,
            'expert_bias_abs_max': biases,
            'global_batch': config.global_batch_size, 'micro_batch': config.micro_batch_size,
            'optimizer': config.optimizer, 'fp8': config.fp8, 'fp8_recipe': config.fp8_recipe,
            'train_iters': config.train_iters, 'lr_warmup_iters': config.lr_warmup_iters,
            'lr_decay_iters': config.lr_decay_iters, 'lr_wsd_decay_iters': config.lr_wsd_decay_iters,
            'data_manifest_sha256': os.environ['STAGE3_MOE_DATA_MANIFEST_SHA256'],
            'bundle_id': os.environ.get('BUNDLE_ID'),
            'scope': 'Fresh model and newly constructed optimizer; no checkpoint load. Hash covers named model parameters before training.',
        }
        Path(os.environ['STAGE3_MOE_COLDSTART_RECEIPT']).write_text(json.dumps(row, sort_keys=True) + '\n')
        print('COLDSTART_INITIALIZATION=' + json.dumps(row, sort_keys=True), flush=True)
        return models, optimizer, scheduler

    training.setup_model_and_optimizer = setup
