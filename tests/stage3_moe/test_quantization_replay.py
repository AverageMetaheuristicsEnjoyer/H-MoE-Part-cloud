import json
from pathlib import Path
import sys

import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('triton')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.quantization_diagnostics import roundtrip
from tools.quantization_replay import TensorReplay, advance, ademamix_direction, install_replay
from tools.dense_replay_source.ademamix import AdEMAMix


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_ademamix_reference_matches_actual_optimizer_with_warmup(tmp_path):
    p=torch.nn.Parameter(torch.linspace(-.1,.1,259,device='cuda'))
    opt=AdEMAMix([p],lr=.003,betas=(.9,.97,.999),alpha=8,beta3_warmup=100,alpha_warmup=100,weight_decay=0)
    p.grad=torch.cos(p.detach());opt.step()
    group=opt.param_groups[0];state=opt.state[p]
    replay=TensorReplay({k:v for k,v in state.items() if torch.is_tensor(v)},group,state['step'],'ademamix',{},tmp_path/'a.jsonl')
    for i in range(8):
        gradient=torch.sin(torch.arange(259,device='cuda')*.01+i*.2)
        before=p.detach().clone();p.grad=gradient.clone()
        replay.update(gradient,group,8);opt.step()
        for key,value in replay.reference.items():
            assert torch.equal(value,opt.state[p][key])
        direction=ademamix_direction(replay.reference,group,replay.step)
        assert torch.allclose(p,before-.003*direction,atol=1e-8,rtol=1e-6)
        protected=replay.variants['dre_fp32_exp_avg_slow']['state']['exp_avg_slow']
        assert torch.equal(protected,replay.reference['exp_avg_slow'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_replay_is_recurrent_not_independent_snapshot(tmp_path):
    values={'exp_avg':torch.linspace(.001,.2,259,device='cuda'), 'exp_avg_sq':torch.linspace(.0001,.02,259,device='cuda')}
    group={'betas':(.9,.99),'eps':1e-8,'lr':.001}
    replay=TensorReplay(values,group,20,'adamw',{},tmp_path/'b.jsonl')
    manual={k:roundtrip(v,k,'maxabs')[0] for k,v in values.items()}
    for i in range(8):
        gradient=torch.cos(torch.arange(259,device='cuda')*.03+i*.1)*.01
        replay.update(gradient,group,8)
        advance(manual,gradient,group,21+i,'adamw')
        manual={k:roundtrip(v,k,'maxabs')[0] for k,v in manual.items()}
    assert all(torch.equal(manual[k],replay.variants['maxabs']['state'][k]) for k in manual)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('kind',['adamw','muon'])
def test_replay_hook_does_not_modify_actual_updates(tmp_path,kind):
    initial=torch.linspace(-.2,.2,768,device='cuda').reshape(32,24)
    p,q=[torch.nn.Parameter(initial.clone()) for _ in range(2)]
    if kind=='adamw':
        make=lambda x:torch.optim.AdamW([x],lr=.003,betas=(.9,.95),weight_decay=.1)
    else:
        from stage3_moe.muon import SPLIT_SWIGLU_FC1, SplitSwiGLUTensorParallelMuon
        make=lambda x:SplitSwiGLUTensorParallelMuon([{'params':[x],SPLIT_SWIGLU_FC1:True}],lr=.003,momentum=.95,nesterov=True,weight_decay=.1,split_qkv=False,fp32_matmul_prec='medium',coefficient_type='quintic',num_ns_steps=5,scale_mode='spectral',extra_scale_factor=.2,pg_collection=None)
    base,shadow=make(p),make(q)
    for opt,param in ((base,p),(shadow,q)):
        param.grad=torch.cos(initial*17);opt.step()
    output=tmp_path/'c.jsonl';install_replay(shadow,[],output,steps=4,samples=1)
    for i in range(5):
        p.grad=torch.sin(initial*(13+i));q.grad=p.grad.clone();base.step();shadow.step()
        assert torch.equal(p,q)
        for key,value in base.state[p].items():
            assert torch.equal(value,shadow.state[q][key]) if torch.is_tensor(value) else value==shadow.state[q][key]
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    assert sum(r['kind']=='replay_block_complete' for r in rows)==1
    assert max(r.get('local_step',0) for r in rows)==4
