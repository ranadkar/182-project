"""Masked, paired BF16 LoRA SFT. Each optimizer step is normalized by its supervised-token count."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5ForCausalLM, set_seed

from protocol import ARMS, HERE, MODEL, branch_prompt, dump, jsonl, tokenizer

CFG=json.loads((HERE/'config.json').read_text())
def backbone():
    torch.set_num_threads(4)
    model, info = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map={'':'cuda:0'},local_files_only=True,output_loading_info=True)
    if info['missing_keys'] or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError(str(info)[:2000])
    assert len(model.model.layers)==CFG['decoder_layers'] and model.config.hidden_size==2560
    assert model.get_output_embeddings().weight.data_ptr()==model.get_input_embeddings().weight.data_ptr()
    model.config.use_cache=False
    return model

def encode(row,arm):
    p=tokenizer().encode(branch_prompt(row,arm),add_special_tokens=False)
    t=row['target_ids']
    pad=row['matched_length']-len(p)-len(t)
    assert pad>=0 and len(t)>1
    ids=p+t+[tokenizer().pad_token_id]*pad
    labels=[-100]*len(p)+t+[-100]*pad
    mask=[1]*(len(p)+len(t))+[0]*pad
    assert [v for v in labels if v!=-100]==t
    return (torch.tensor(ids)[None,:],torch.tensor(mask)[None,:],torch.tensor(labels)[None,:])

def loss(model,batch,reference=False):
    ids,mask,labels=[v.to('cuda') for v in batch]
    which=(labels[0]!=-100).nonzero().flatten()
    kwargs=dict(input_ids=ids,attention_mask=mask,use_cache=False)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(**kwargs,logits_to_keep=which-1).logits
        value=torch.nn.functional.cross_entropy(logits[0].float(),labels[0,which],reduction='sum')
        if reference:
            ordinary=model(**kwargs,labels=labels).loss
            torch.testing.assert_close(value/which.numel(),ordinary.float(),rtol=0.003,atol=0.003)
    return value

@torch.no_grad()
def validation(model, data):
    model.eval()
    val=sum(float(loss(model,d)) for d in data)/sum(int((d[2]!=-100).sum()) for d in data)
    model.train()
    return val

def digest_params(m):
    h=hashlib.sha256()
    for name,p in m.named_parameters():
        if p.requires_grad:
            h.update(name.encode()); h.update(p.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()

def package_adapter(out):
    cfg_path=out/'adapter_config.json'
    metadata=json.loads(cfg_path.read_text())
    metadata.update(base_model_name_or_path=CFG['checkpoint'],revision=CFG['revision'])
    dump(cfg_path,metadata)
    (out/'README.md').write_text(
        f"# {out.name}\n\n"
        f"PEFT LoRA adapter for the text decoder of {CFG['checkpoint']} at revision "
        f"`{CFG['revision']}`. The fitted parameters are in `adapter_model.safetensors`. "
        "The training and validation measurements are in [run.json](run.json) and [loss.jsonl](loss.jsonl). "
        "See the [experiment report](../../report.md) for the intervention and held-out outcomes.\n")

def fit(arm,seed,pilot=False):
    out=HERE/'adapters'/f'{"pilot" if pilot else "main"}-{arm}-{seed}'
    if (out/'run.json').exists(): raise FileExistsError(out)
    out.mkdir(exist_ok=True)
    set_seed(seed)
    model=backbone()
    options=LoraConfig(r=CFG['rank'],lora_alpha=CFG['alpha'],lora_dropout=CFG['dropout'],
        target_modules=['gate_proj','up_proj','down_proj'],layers_to_transform=list(range(CFG['first_adapter_layer'],CFG['decoder_layers'])),
        layers_pattern='layers',bias='none',task_type='CAUSAL_LM')
    model=get_peft_model(model,options)
    trainables={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert len(trainables)==(CFG['decoder_layers']-CFG['first_adapter_layer'])*3*2
    assert all('.mlp.' in n and '.lora_' in n for n in trainables)
    initial=digest_params(model)
    train_rows=jsonl(HERE/'data/training.jsonl')[:CFG['pilot_examples'] if pilot else CFG['main_examples']]
    val_rows=jsonl(HERE/'data/validation.jsonl')[:8 if pilot else None]
    encoded=[encode(r,arm) for r in train_rows]
    val=[encode(r,arm) for r in val_rows]
    steps=len(encoded)//CFG['accumulation']*CFG['epochs']
    assert len(encoded)%CFG['accumulation']==0
    optimizer=torch.optim.AdamW(trainables.values(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
    warmup=max(1,math.ceil(steps*CFG['warmup_fraction']))
    order=list(range(len(encoded))); random.Random(seed).shuffle(order)
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    model.eval()
    with torch.no_grad(): loss(model,encoded[0],reference=True)
    before=validation(model,val); total_tokens=0; history=[]
    set_seed(seed)
    with (out/'loss.jsonl').open('w') as f:
        for step in range(steps):
            lr=CFG['lr']*min((step+1)/warmup,(steps-step)/max(1,steps-warmup))
            optimizer.param_groups[0]['lr']=lr
            ix=[order[(step*CFG['accumulation']+j)%len(order)] for j in range(CFG['accumulation'])]
            tokens=sum(len(train_rows[i]['target_ids']) for i in ix)
            objective=torch.zeros((),device='cuda')
            for i in ix:
                summed=loss(model,encoded[i])
                (summed/tokens).backward()
                objective+=summed.detach()
            grad=torch.nn.utils.clip_grad_norm_(trainables.values(),1.0,error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total_tokens+=tokens
            rec=dict(step=step+1,lr=lr,train_nll=float(objective)/tokens,grad_norm=float(grad),target_tokens=tokens,
                     ids=[train_rows[i]['id'] for i in ix],elapsed_s=round(time.monotonic()-start,3))
            if step+1 in {max(1,steps//2),steps}: rec['validation_nll']=validation(model,val)
            history.append(rec)
            f.write(json.dumps(rec)+'\n'); f.flush()
            if step in {0,steps-1} or (step+1)%8==0: print(out.name,step+1,rec,flush=True)
    final=digest_params(model)
    assert initial!=final and total_tokens==sum(len(r['target_ids']) for r in train_rows)*CFG['epochs']
    model.save_pretrained(out,save_embedding_layers=False)
    package_adapter(out)
    dump(out/'run.json',dict(config=CFG,arm=arm,seed=seed,pilot=pilot,total_target_tokens=total_tokens,
        initialization_sha256=initial,final_sha256=final,trainable_parameters=sum(p.numel() for p in trainables.values()),
        modules=list(trainables),gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda=torch.version.cuda,
        wall_s=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,validation_before=before,validation_after=history[-1]['validation_nll'],
        trainable_gradient_clip=1.,masked_reference_verified=True))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('arm',choices=ARMS); p.add_argument('seed',type=int); p.add_argument('--pilot',action='store_true')
    a=p.parse_args(); fit(a.arm,a.seed,a.pilot)
