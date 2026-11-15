"""AdEMAMix state replay on real gradients of a frozen Dense checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/dense_replay_source'))
from models.llama import Llama
from megatron.core.datasets.indexed_dataset import IndexedDataset
from tools.quantization_diagnostics import emit, materialize, selected_parameters
from tools.quantization_replay import TensorReplay


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=128)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    weights=checkpoint['model']
    for prefix in ('module.','_orig_mod.'):
        if all(k.startswith(prefix) for k in weights):weights={k[len(prefix):]:v for k,v in weights.items()}
    config=SimpleNamespace(vocab_size=50304,sequence_length=1024,n_embd=1280,n_head=20,n_layer=18,
                           multiple_of=256,rmsnorm_eps=1e-5,init_std=0.02,dropout=0.0,fp8=False)
    torch.manual_seed(1234)
    model=Llama(config)
    model.load_state_dict(weights,strict=True)
    model=model.cuda().eval()
    named=dict(model.named_parameters())
    specs=model.get_parameter_group_specs()
    saved=checkpoint['optimizer']
    assert len(specs)==len(saved['param_groups'])
    mapping={}
    for spec,group in zip(specs,saved['param_groups']):
        assert len(spec['params'])==len(group['params'])
        for name,pid in zip(spec['params'],group['params']):
            mapping[pid]=name
            assert tuple(saved['state'][pid]['exp_avg_sq'].shape)==tuple(named[name].shape)
    selected=list(selected_parameters(saved['param_groups'],saved['state'],3))
    selected_names={mapping[p] for _,_,p,_ in selected}
    for name,p in named.items():p.requires_grad_(name in selected_names)
    replays=[]
    for gi,pos,pid,keys in selected:
        state=saved['state'][pid];group=saved['param_groups'][gi]
        assert group['betas'][0]>0
        values={k:materialize(state,k) for k in ('exp_avg_fast','exp_avg_slow','exp_avg_sq')}
        label={'optimizer':'AdEMAMix','group_index':gi,'position':pos,'name':mapping[pid]}
        replay=TensorReplay(values,group,int(state['step']),'ademamix',label,args.output)
        replays.append((named[mapping[pid]],group,replay))
    prefixes=[Path(p)/'train' for p in ('/home/jovyan/data/fineweb-edu-gpt2-megatron/data',
              '/workspace-SR006.nfs2/hmoe-data/fineweb-edu-gpt2-megatron/data',
              '/workspace-SR006.nfs3/hmoe-data/fineweb-edu-gpt2-megatron/data')]
    prefix=next(p for p in prefixes if p.with_suffix('.bin').is_file())
    dataset=IndexedDataset(str(prefix))
    emit(args.output,{'kind':'protocol','mode':'frozen_dense_checkpoint_real_gradient_replay','checkpoint_iteration':checkpoint['itr'],
         'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
         'steps':args.steps,'model_config':vars(config),'micro_batch':8,'gradient_accumulation':8,'batch_sequences':64,
         'compute':'BF16 autocast, FP32 model and replay state arithmetic','parameter_mapping':'source sorted decay/no-decay groups; strict model and all state shapes validated',
         'data':str(prefix),'data_cursor':'fresh sequential documents from index zero, packed across boundaries',
         'limitation':'not historical training-stream resume; gradients from fixed weights; current MoE codecs applied to Dense AdEMAMix states; no causal claim about historical failure'})
    document=0;pending=np.empty(0,dtype=np.int64);token_hash=hashlib.sha256()
    parameters=[p for p,_,_ in replays]
    for local_step in range(1,args.steps+1):
        model.zero_grad(set_to_none=True);loss_sum=0.0
        for micro in range(8):
            needed=8*1025
            while len(pending)<needed:
                pending=np.concatenate((pending,np.asarray(dataset[document],dtype=np.int64)));document+=1
            batch=pending[:needed].reshape(8,1025).copy();pending=pending[needed:]
            token_hash.update(batch.tobytes())
            ids=torch.from_numpy(batch).cuda()
            assert int(ids.max())<config.vocab_size and int(ids.min())>=0
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(ids[:,:-1].contiguous(),ids[:,1:].contiguous())['loss']
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite Dense replay loss')
            (loss/8).backward();loss_sum+=float(loss)/8
        # Clipping is intentionally omitted: only selected tensor gradients exist.
        for parameter,group,replay in replays:replay.update(parameter.grad,group,args.steps)
        if local_step in (1,2,4,8,16,32,64,128) or local_step==args.steps:
            emit(args.output,{'kind':'gradient_batch','local_step':local_step,'loss':loss_sum,'tokens_sha256_prefix':token_hash.hexdigest(),'documents_read':document,'gradient_clip':'none; sampled-gradient diagnostic'})
            print(f'ADEMAMIX_REPLAY_STEP={local_step}/{args.steps} LOSS={loss_sum}',flush=True)
    emit(args.output,{'kind':'replay_complete','steps':args.steps,'tensors':len(replays),'tokens_sha256':token_hash.hexdigest(),'model_weights':'unchanged'})
    print('ADEMAMIX_REPLAY=PASS',flush=True)


if __name__=='__main__':main()
