"""Native-fused, inference-only RWKV state. No change to training code/weights.

The new WKV kernel preserves @rwkv3's arithmetic but accepts/returns FP32 state.
Native BF16 mix/gate/GroupNorm/channel-mix functions are reused. Split GEMM shapes
can cause BF16 differences: audits quantify these, never promise bitwise parity.
"""
from __future__ import annotations
from pathlib import Path
import torch

_EXTENSION_LOADED=False


def load_stateful_wkv():
    global _EXTENSION_LOADED
    if not _EXTENSION_LOADED:
        from torch.utils.cpp_extension import load
        load(name='rosa_rwkv7_stateful_v1',sources=[str(Path(__file__).parent/'cuda/stateful_wkv.cu')],
             extra_cflags=['-O3'],extra_cuda_cflags=['--use_fast_math','-O3','-Xptxas','-O3'],
             is_python_module=False,verbose=True)
        _EXTENSION_LOADED=True
    return torch.ops.rosa_rwkv7_stateful.forward


def wkv_stateful(r,w,k,v,a,b,state=None,state_tokens=None):
    B,T,C=r.shape;H=C//64
    if C%64:raise ValueError('Native RWKV head size must be 64.')
    if state is None:state=torch.zeros((B,H,64,64),device=r.device,dtype=torch.float32)
    inputs=[x.reshape(B,T,H,64).contiguous() for x in (r,w,k,v,a,b)]
    out,final=load_stateful_wkv()(*inputs,state.contiguous(),T if state_tokens is None else state_tokens)
    return out.reshape(B,T,C),final


class StatefulNative:
    def __init__(self,body):
        if hasattr(body,'rosa_cpu'):raise ValueError('Post-norm adapter is applied separately, not within frozen body.')
        from src import model as native
        self.body=body;self.native=native
        load_stateful_wkv()

    @torch.inference_mode()
    def features(self,ids,state=None):
        if ids.ndim!=2 or ids.shape[1]<1 or ids.device.type!='cuda':
            raise ValueError('Expected nonempty CUDA token chunk.')
        if state is not None and len(state)!=len(self.body.blocks):raise ValueError('Layer-state count mismatch.')
        valid_tokens=ids.shape[1]
        # Native prefill uses tail padding. Keep exactly the same GEMM dimensions
        # through all layers, but snapshot recurrence BEFORE those causal pads.
        if state is None:ids=torch.nn.functional.pad(ids,(0,(-valid_tokens)%16),value=65532)
        x=self.body.emb(ids);v_first=torch.empty_like(x);new=[];n=self.native
        for layer,block in enumerate(self.body.blocks):
            previous=None if state is None else state[layer]
            if layer==0:x=block.ln0(x)
            att_x=block.ln1(x);att=block.att
            mixed_input=att_x if previous is None else torch.cat((previous['att_x'].unsqueeze(1),att_x),dim=1)
            mixed=n.tmix_mix6_bf16_v5(mixed_input,att.x_r.view(-1),att.x_w.view(-1),att.x_k.view(-1),
                                     att.x_v.view(-1),att.x_a.view(-1),att.x_g.view(-1))
            if previous is not None:mixed=tuple(y[:,1:].contiguous() for y in mixed)
            xr,xw,xk,xv,xa,xg=mixed
            r=att.receptance(xr);w=att.w0+torch.tanh(xw@att.w1)@att.w2
            k=att.key(xk);v=att.value(xv)
            if layer==0:v_first=v
            else:v=n.tmix_vres_gate_bf16_v1(v,v_first,att.v0,(xv@att.v1)@att.v2)
            a=n.tmix_a_gate_bf16(att.a0,(xa@att.a1)@att.a2)
            g=torch.sigmoid(xg@att.g1)@att.g2
            k,negative_kk,kka=n.tmix_kk_pre_bf16_v5(k,att.k_k.view(-1),a,att.k_a.view(-1))
            y,wkv=wkv_stateful(r,w,k,v,negative_kk,kka,None if previous is None else previous['wkv'],state_tokens=valid_tokens)
            y=n.tmix_lnx_rkvres_xg_bf16_v1(y,r,k,v,att.r_k,att.ln_x.weight,att.ln_x.bias,g)
            x=x+att.output(y)
            ffn_x=block.ln2(x)
            ffn_input=ffn_x if previous is None else torch.cat((previous['ffn_x'].unsqueeze(1),ffn_x),dim=1)
            ffn_out=block.ffn(ffn_input)
            if previous is not None:ffn_out=ffn_out[:,1:]
            x=x+ffn_out
            new.append({'att_x':att_x[:,valid_tokens-1].clone(),'ffn_x':ffn_x[:,valid_tokens-1].clone(),'wkv':wkv})
        return self.body.ln_out(x)[:,:valid_tokens],new


def state_nbytes(state):
    return sum(t.numel()*t.element_size() for layer in state for t in layer.values())


class IncrementalAddress:
    """Same integer policy as training, persistent compiled CPU SAM/LCT state."""
    SPECIALS={0,65530,65531,65532,65533,65534}

    def __init__(self):
        from rosa_cpp_extension import load_rosa_sam_cpu_extension
        self.state=load_rosa_sam_cpu_extension().IncrementalSAM(1)
        self.last=None

    @staticmethod
    def pack(predictions,lengths):
        from rosa_runtime import RosaAddressBatch
        p=torch.tensor([predictions],dtype=torch.long)
        raw=torch.tensor([lengths],dtype=torch.long)
        special=torch.zeros_like(p,dtype=torch.bool)
        for token in IncrementalAddress.SPECIALS:special|=p.eq(token)
        valid=p.ge(0)&raw.ge(1)&~special
        return RosaAddressBatch(addr_ids=torch.where(valid,p,torch.full_like(p,-1)),
            raw_match_lens=raw,fired_match_lens=torch.where(valid,raw,torch.zeros_like(raw)),
            valid_mask=valid,special_mask=special,source='incremental:compiled_sam_lct_v1')

    def prefill(self,ids):
        if not ids:raise ValueError('Empty SAM prefill.')
        p,m=self.state.prefill(list(ids));self.last=self.pack(p[-1:],m[-1:])
        return self.pack(p,m)

    def update_one(self,token):
        p,m=self.state.update_one(int(token));self.last=self.pack([p],[m]);return self.last

    def clone(self):
        other=object.__new__(type(self));other.state=self.state.clone();other.last=self.last;return other
