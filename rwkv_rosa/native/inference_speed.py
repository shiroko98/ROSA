"""Batch-one128K inference timings for GPU, hybrid and CPU placements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import time
import traceback

import numpy as np
import torch

from .adapter import CPUImprovedROSA
from .cpu_compare import CPUInference, state_hash, write_json
from .cpu_wkv_inference import install_cpu_wkv
from .gpu_needle import NativeGPU
from .gpu_state import StatefulNative, IncrementalAddress
from .needle_eval import build_case, clean_decode
from .position import scale_options
from rwkv_rosa.backbone import load_base_state, build_backbone, infer_config
from third_party.rwkv_lm_v7.tokenizer import TRIE_TOKENIZER

ROOT=Path(__file__).resolve().parents[2]


class Runner:
    def __init__(self,manifest,placement,folder):
        self.placement=placement;self.folder=folder;self.device='cpu' if placement=='cpu' else 'cuda'
        self.meta=json.loads(Path(manifest['adapter_snapshot']).with_suffix('.json').read_text())
        weights=torch.load(manifest['adapter_snapshot'],map_location='cpu',weights_only=True,mmap=True)
        assert state_hash(weights)==self.meta['adapter_state_sha256']
        self.tokenizer=TRIE_TOKENIZER(str(ROOT/'rwkv_vocab_v20230424.txt'))
        if placement=='cpu':
            base=load_base_state(self.meta['base_model'])
            assert state_hash(base,self.meta['base_parameter_names'])==self.meta['base_sha256_native_order']
            self.body=build_backbone(infer_config(base),dtype=torch.float32,state_dict=base)
            del base
            self.runtime=CPUInference(self.body)
            probe=torch.tensor([[1,2,3,4,5,6,7,8]*8])
            expected,_=self.runtime.features(probe,return_state=True)
            install_cpu_wkv(self.body)
            actual,_=self.runtime.features(probe,return_state=True)
            torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-4)
            self.audit={'cpu_compiled_reference_max_hidden_error':float((actual-expected).abs().max())}
            self.adapter=CPUImprovedROSA(self.body.emb,position='post_norm',device='cpu',
                address_impl='compiled_cpu',scale=self.meta['scale'],output_norm=self.meta['output_norm'],**scale_options(self.meta))
            self.adapter.load_state_dict(weights,strict=True)
        else:
            self.native=NativeGPU(manifest['train_dir'],manifest['adapter_snapshot'],136192)
            self.body=self.native.body;self.runtime=StatefulNative(self.body)
            self.audit={'gpu':torch.cuda.get_device_name()}
            if placement=='gpu':self.adapter=self.native.adapter
            else:
                embedding=torch.nn.Embedding.from_pretrained(self.body.emb.weight.detach().float().cpu(),freeze=True)
                self.adapter=CPUImprovedROSA(embedding,position='post_norm',device='cpu',address_impl='compiled_cpu',
                    scale=self.meta['scale'],output_norm=self.meta['output_norm'],**scale_options(self.meta))
                self.adapter.load_state_dict(weights,strict=True)
                # The GPU adapter allocated by the strict loader is not used by
                # this placement. Release it before measuring memory/latency.
                self.native.adapter=None
                del embedding
                torch.cuda.empty_cache()
        self.body.eval().requires_grad_(False);self.adapter.eval().requires_grad_(False)
        self.adapter.bind_output_head(self.body.head)
        assert all(p.device.type==self.device for p in self.body.parameters())
        target='cuda' if placement=='gpu' else 'cpu'
        assert all(p.device.type==target and p.dtype==torch.float32 for p in self.adapter.parameters())
        assert not any(p.requires_grad for p in (*self.body.parameters(),*self.adapter.parameters()))
        self.body_parameters=sum(p.numel() for p in self.body.parameters())
        self.adapter_parameters=sum(p.numel() for p in self.adapter.parameters())

    def sync(self):
        if self.device=='cuda':torch.cuda.synchronize()

    def features(self,ids,state=None):
        tokens=torch.tensor([ids],device=self.device,dtype=torch.long)
        if self.device=='cpu':return self.runtime.features(tokens,state=state,return_state=True)
        return self.runtime.features(tokens,state)

    def predict(self,h,address):
        self.adapter.addresses=address
        logits=self.body.head(self.adapter.inject_post_norm(h[:,-1:])).float()[0,-1]
        if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Nonfinite benchmark logits')
        return int(logits.argmax()),logits

    def decode(self,h,state,sam,count,record=True):
        # First token belongs to TTFT. Time count subsequent sequential forward
        # and prediction steps, including SAM, transfers and host token selection.
        address=sam.clone();first,_=self.predict(h,address.last);self.sync()
        token=first;latencies=[];ids=[];active=0;gains=[]
        began=time.perf_counter()
        for index in range(count):
            start=time.perf_counter()
            hidden,state=self.features([token],state)
            last=address.update_one(token)
            token,_=self.predict(hidden,last)
            self.sync()
            latencies.append(time.perf_counter()-start);ids.append(token)
            if record:
                rms=self.adapter.stats.get('rosa_residual_rms',0.)
                gain=self.adapter.stats.get('rosa_dynamic_scale_max',0.)
                if int(last.fired_match_lens[0,0])<6:assert rms==gain==0
                assert 0<=gain<=.300001
                active+=rms>0;gains.append(gain)
        seconds=time.perf_counter()-began
        return dict(tokens=count,decode_seconds=seconds,tokens_per_second=count/seconds,
            mean_ms_per_token=1000*seconds/count,p50_ms_per_token=1000*statistics.median(latencies),
            p95_ms_per_token=1000*float(np.quantile(latencies,.95)),
            rosa_nonzero_steps=active,first_token=first,generated_ids=ids,
            generated_text=clean_decode(self.tokenizer,[first]+ids),latency_seconds=latencies,
            effective_gains=gains,decode_policy='Greedy,fixed-step throughput benchmark; EOS recorded but does not shorten the timed loop.',
            stop_tokens_seen=sum(t in (0,65531,65532) for t in ids))


@torch.inference_mode()
def benchmark(folder,placement):
    m=json.loads((folder/'manifest.json').read_text());out=folder/placement;out.mkdir(exist_ok=True)
    def progress(phase,**kw):
        write_json(out/'status.json',dict(status='running',phase=phase,pid=os.getpid(),**kw))
    os.sched_setaffinity(0,set(range(48)))
    torch.set_num_interop_threads(1);torch.set_num_threads(48 if placement=='cpu' else 4)
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    progress('loading')
    load_start=time.perf_counter();r=Runner(m,placement,folder);load_seconds=time.perf_counter()-load_start
    warm=build_case(r.tokenizer,1024,.5,m['seed'],'similar_archives',prompt_template='rwkv')
    warm_ids=warm['stem_ids']+warm['suffix_ids']['exact_prefix']
    warm_h,warm_state=r.features(warm_ids);warm_sam=IncrementalAddress();warm_sam.prefill(warm_ids)
    _,warm_logits=r.predict(warm_h,warm_sam.last);r.sync()
    if placement=='gpu':torch.save(warm_logits.cpu(),folder/'gpu_warm_logits.pt')
    elif placement=='hybrid':
        ref=torch.load(folder/'gpu_warm_logits.pt',weights_only=True,map_location='cpu')
        error=float((warm_logits.cpu()-ref).abs().max())
        torch.testing.assert_close(warm_logits.cpu(),ref,atol=.002,rtol=.002)
        r.audit.update(hybrid_gpu_max_logit_error=error,hybrid_gpu_top1_equal=int(warm_logits.argmax())==int(ref.argmax()))
    r.decode(warm_h,warm_state,warm_sam,8,record=False)
    tuning={};prefill_threads=4;decode_threads=4
    if placement=='cpu':
        progress('thread_tuning');prefill=[];decode=[]
        for threads in [8,16,32,48]:
            torch.set_num_threads(threads)
            r.decode(warm_h,warm_state,warm_sam,2,record=False)
            result=r.decode(warm_h,warm_state,warm_sam,8,record=False)
            decode.append(dict(threads=threads,tokens_per_second=result['tokens_per_second']))
        for threads in [16,32,48]:
            torch.set_num_threads(threads);r.features(warm_ids[:256])
            start=time.perf_counter();r.features(warm_ids[:1024]);seconds=time.perf_counter()-start
            prefill.append(dict(threads=threads,seconds=seconds))
        decode_threads=max(decode,key=lambda x:x['tokens_per_second'])['threads']
        prefill_threads=min(prefill,key=lambda x:x['seconds'])['threads']
        tuning=dict(decode=decode,prefill=prefill,selected_decode_threads=decode_threads,selected_prefill_threads=prefill_threads,
            note='Short warm-up calibration on one48-core NUMA node,not a universal tuning result.')
        write_json(out/'thread_tuning.json',tuning)
    torch.set_num_threads(prefill_threads)
    case=build_case(r.tokenizer,131072,.5,m['seed'],'similar_archives',prompt_template='rwkv')
    write_json(out/'input.json',{k:v for k,v in case.items() if k not in ('stem_ids','suffix_ids')})
    if r.device=='cuda':torch.cuda.reset_peak_memory_stats()
    progress('document_prefill',processed_tokens=0,total_tokens=131072,prefill_threads=prefill_threads,decode_threads=decode_threads)
    state=None;hidden=None;chunk=m['cpu_prefill_chunk'] if placement=='cpu' else len(case['stem_ids'])
    r.sync();start=time.perf_counter()
    for at in range(0,len(case['stem_ids']),chunk):
        hidden,state=r.features(case['stem_ids'][at:at+chunk],state)
        if placement=='cpu':progress('document_prefill',processed_tokens=min(at+chunk,131072),total_tokens=131072,prefill_threads=prefill_threads,decode_threads=decode_threads)
    r.sync();body_prefill=time.perf_counter()-start
    del hidden
    start=time.perf_counter();doc_sam=IncrementalAddress();doc_sam.prefill(case['stem_ids']);sam_prefill=time.perf_counter()-start
    results=[]
    for mode in ['qa','exact_prefix']:
        # Share the immutable document state across suffixes. No output token
        # from another repetition or prompt is inserted in this state or SAM.
        torch.set_num_threads(prefill_threads);r.sync();start=time.perf_counter()
        h,prompt_state=r.features(case['suffix_ids'][mode],state);r.sync();suffix_body=time.perf_counter()-start
        sam=doc_sam.clone();start=time.perf_counter()
        for token in case['suffix_ids'][mode]:sam.update_one(token)
        suffix_sam=time.perf_counter()-start
        torch.set_num_threads(decode_threads);start=time.perf_counter();r.predict(h,sam.last);r.sync();initial_prediction=time.perf_counter()-start
        ttft=body_prefill+sam_prefill+suffix_body+suffix_sam+initial_prediction
        prompt_hash=hashlib.sha256(np.array(case['stem_ids']+case['suffix_ids'][mode],dtype=np.int64).tobytes()).hexdigest()
        for repeat in range(m['repetitions']):
            progress('decode',mode=mode,repeat=repeat+1,repetitions=m['repetitions'])
            row=r.decode(h,prompt_state,sam,m['decode_tokens'])
            row.update(placement=placement,mode=mode,repeat=repeat,document_tokens=131072,
                prompt_tokens=131072+len(case['suffix_ids'][mode]),prompt_sha256=prompt_hash,
                document_body_prefill_seconds=body_prefill,document_sam_prefill_seconds=sam_prefill,
                suffix_body_seconds=suffix_body,suffix_sam_seconds=suffix_sam,initial_prediction_seconds=initial_prediction,
                estimated_ttft_seconds=ttft,prefill_tokens_per_second=131072/(body_prefill+sam_prefill),
                prefill_threads=prefill_threads,decode_threads=decode_threads)
            results.append(row);write_json(out/'measurements.json',results)
            print('MEASURE',json.dumps({k:row[k] for k in ['placement','mode','repeat','tokens_per_second','estimated_ttft_seconds','rosa_nonzero_steps']}),flush=True)
    report=dict(status='complete',placement=placement,measurements=results,thread_tuning=tuning,
        audit=r.audit,base_sha256=r.meta['base_sha256_native_order'],adapter_sha256=r.meta['adapter_state_sha256'],
        body_parameters=r.body_parameters,adapter_parameters=r.adapter_parameters,load_seconds_excluded=load_seconds,
        cpu_affinity=sorted(os.sched_getaffinity(0)),max_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        peak_gpu_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if r.device=='cuda' else 0,
        body_dtype=str(next(r.body.parameters()).dtype),adapter_dtype='torch.float32',
        training=False,benchmark_fixed_steps=True,prefill_repetitions=1,
        ttft_note='Warm loaded model; one measured document prefill per placement,shared across prompts/repeats. TTFT=sum(document body+SAM,suffix body+SAM,initial prediction). Excludes tokenization,weight loading,JIT,warmup and repeat-fork clone.',
        addressing='Compiled incremental SAM/LCT on CPU in all placements; no full-history rebuilding during decode.',
        hybrid_note='CPU trainable ROSA tensors; bounded dynamic-gain controller uses GPU frozen head; transfers and synchronization included.')
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='complete'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--placement',choices=['gpu','hybrid','cpu'],required=True);a=p.parse_args()
    folder=Path(a.output)
    try:benchmark(folder,a.placement)
    except Exception:
        child=folder/a.placement;child.mkdir(exist_ok=True);write_json(child/'failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
