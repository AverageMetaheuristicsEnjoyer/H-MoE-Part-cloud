"""Recurrent state replay driven by a shared sequence of real gradients."""
import os
from pathlib import Path

import torch

from tools.quantization_diagnostics import (
    SIGNED, adam_direction, emit, master_parameter_names, materialize, metrics,
    raw_optimizers, roundtrip, selected_parameters,
)
from tools.dense_replay_source.ademamix import (
    linear_hl_warmup_scheduler, linear_warmup_scheduler,
)


def advance(values, gradient, group, step, kind):
    if kind == 'muon':
        values['momentum_buffer'].lerp_(gradient, 1 - group['momentum'])
        return
    if kind == 'ademamix':
        beta1, beta2, final_beta3 = group['betas']
        beta3 = linear_hl_warmup_scheduler(step, final_beta3, beta1, group['beta3_warmup']) if group['beta3_warmup'] is not None else final_beta3
        values['exp_avg_fast'].mul_(beta1).add_(gradient, alpha=1-beta1)
        values['exp_avg_slow'].mul_(beta3).add_(gradient, alpha=1-beta3)
        values['exp_avg_sq'].mul_(beta2).addcmul_(gradient, gradient, value=1-beta2)
        return
    beta1, beta2 = group['betas']
    values['exp_avg'].lerp_(gradient, 1-beta1)
    values['exp_avg_sq'].mul_(beta2).addcmul_(gradient, gradient, value=1-beta2)


def ademamix_direction(values, group, step):
    beta1, beta2, _ = group['betas']
    alpha = linear_warmup_scheduler(step, group['alpha'], 0, group['alpha_warmup']) if group['alpha_warmup'] is not None else group['alpha']
    return (values['exp_avg_fast'] / (1-beta1**step) + alpha*values['exp_avg_slow']) / (values['exp_avg_sq'].sqrt() / (1-beta2**step)**0.5 + group['eps'])


class TensorReplay:
    def __init__(self, values, group, step, kind, label, output, direction_fn=None):
        self.kind, self.label, self.output = kind, label, output
        self.step, self.local_step = step, 0
        self.reference = {k:v.clone() for k,v in values.items()}
        self.direction_fn = direction_fn
        self.variants = {}
        plans = [('maxabs', ()), ('dre', ())]
        if kind == 'ademamix':
            plans += [('dre', (name,)) for name in ('exp_avg_fast','exp_avg_slow','exp_avg_sq')]
        for codec, protected in plans:
            name = codec + ('_fp32_'+protected[0] if protected else '')
            state = {}; storage = 0
            for key, value in values.items():
                if key in protected:
                    state[key] = value.clone(); storage += value.numel()*4
                else:
                    state[key], info = roundtrip(value,key,codec)
                    storage += info['payload_bytes']+info['metadata_bytes']
            self.variants[name] = {'codec':codec, 'protected':protected, 'state':state,
                                   'lost':{k:0 for k in values}, 'changed':{k:0 for k in values}}
            emit(output, {**label,'kind':'replay_variant','variant':name,'protected':protected,
                          'theoretical_state_bytes':storage,'fp32_state_bytes':sum(v.numel()*4 for v in values.values()),
                          'excludes':'temporary replay buffers and training peak memory'})
        emit(output, {**label,'kind':'replay_tensor','optimizer_kind':kind,'initial_step':step,
                      'shape':list(next(iter(values.values())).shape),
                      'group':{k:v for k,v in group.items() if k!='params' and isinstance(v,(str,int,float,bool,tuple,type(None)))}})

    def direction(self, values, gradient, group):
        if self.kind == 'muon':
            return self.direction_fn(values['momentum_buffer'],gradient,group)
        if self.kind == 'ademamix':
            return ademamix_direction(values,group,self.step)
        return adam_direction(values,group,self.step)

    @torch.no_grad()
    def update(self, gradient, group, total_steps):
        self.local_step += 1; self.step += 1
        log = self.local_step in (1,2,4,8,16,32,64,128) or self.local_step==total_steps
        advance(self.reference,gradient,group,self.step,self.kind)
        reference_direction = self.direction(self.reference,gradient,group) if log else None
        for name, entry in self.variants.items():
            state = entry['state']
            before = {k:v.clone() for k,v in state.items()}
            advance(state,gradient,group,self.step,self.kind)
            if log:
                emit(self.output,{**self.label,'kind':'replay_update_direction','variant':name,
                     'local_step':self.local_step,'optimizer_step':self.step,
                     'excludes':'common learning-rate scaling and decoupled weight decay; not autonomous loss',
                     **metrics(reference_direction,self.direction(state,gradient,group))})
            for key,value in list(state.items()):
                changed = value != before[key]
                decoded = value if key in entry['protected'] else roundtrip(value,key,entry['codec'])[0]
                entry['lost'][key] += int(((decoded==before[key]) & changed).sum())
                entry['changed'][key] += int(changed.sum())
                state[key] = decoded
                if log:
                    emit(self.output,{**self.label,'kind':'replay_state','state':key,'variant':name,
                         'local_step':self.local_step,'optimizer_step':self.step,
                         'cumulative_lost_change_fraction':entry['lost'][key]/max(1,entry['changed'][key]),
                         'meaning':'recurrent decoded state unchanged despite a nonzero update from its own prior state',
                         **metrics(self.reference[key],decoded)})


def install_replay(optimizer, models, output_path, steps=128, samples=3):
    output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True)
    names={}
    for module in models:
        while hasattr(module,'module'):module=module.module
        names.update({id(p):n for n,p in module.named_parameters()})
    master_parameter_names(optimizer,names)
    emit(output,{'kind':'protocol','mode':'recurrent_common_real_gradients','steps':steps,'samples_per_stratum':samples,
                 'selection':'first/middle/last within optimizer group, shape, state keys','group_size':128,
                 'limitation':'gradients follow the FP32-state model; no autonomous quantized-model loss claim'})
    for block,raw in enumerate(raw_optimizers(optimizer)):
        selected=[];counter=[0]
        def hook(current,args,kwargs,block=block,selected=selected,counter=counter):
            if counter[0]>=steps:return
            with torch.no_grad():
                if not selected:
                    for gi,pos,p,keys in selected_parameters(current.param_groups,current.state,samples):
                        if p.grad is None:continue
                        group=current.param_groups[gi];state=current.state[p]
                        values={k:materialize(state,k) for k in SIGNED if k in state and torch.is_tensor(state[k])}
                        kind='muon' if 'momentum_buffer' in values else 'adamw'
                        fn=None
                        if kind=='muon':
                            def fn(buffer,gradient,group,p=p,current=current):
                                incoming=gradient.lerp(buffer,group['momentum']) if current.nesterov else buffer
                                precision=torch.get_float32_matmul_precision()
                                torch.set_float32_matmul_precision(current.fp32_matmul_prec)
                                try:return current.orthogonalize(p,incoming,**{k:v for k,v in group.items() if k!='params'}).float()
                                finally:torch.set_float32_matmul_precision(precision)
                        step=int(state.get('step',group.get('step',0)))
                        label={'block':block,'group_index':gi,'position':pos,'name':names.get(id(p)),'optimizer':type(current).__name__}
                        replay=TensorReplay(values,group,step,kind,label,output,fn)
                        selected.append((p,gi,replay))
                for p,gi,replay in selected:
                    replay.update(p.grad.detach().float(),current.param_groups[gi],steps)
                counter[0]+=1
                if counter[0]==steps:
                    emit(output,{'kind':'replay_block_complete','block':block,'steps':steps,'tensors':len(selected)})
                    selected.clear()
        raw.register_step_pre_hook(hook)
