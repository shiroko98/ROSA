import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder=Path(sys.argv[1]);r=json.loads((folder/'summary.json').read_text());placements=['gpu','hybrid','cpu']
fig,axes=plt.subplots(1,2,figsize=(12,5))
for offset,mode,color,label in [(-.18,'qa','#146c94','Natural QA'),(.18,'exact_prefix','#e69532','Exact-prefix completion')]:
    vals=[next(g for g in r['summary'] if g['placement']==p and g['mode']==mode)['tokens_per_second'] for p in placements]
    bars=axes[0].bar(np.arange(3)+offset,vals,.36,color=color,label=label)
    for b,v in zip(bars,vals):axes[0].text(b.get_x()+b.get_width()/2,v+max(vals)*.025,f'{v:.2f}',ha='center',fontsize=9)
axes[0].set_xticks(range(3),['GPU body\nGPU ROSA','GPU body\nCPU ROSA','CPU body\nCPU ROSA']);axes[0].set_ylabel('Generated tokens / second');axes[0].set_title('Decode: batch1, 3 repeats x128 steps');axes[0].legend(frameon=False)
vals=[next(g for g in r['summary'] if g['placement']==p and g['mode']=='qa')['document_prefill_seconds'] for p in placements]
bars=axes[1].bar(range(3),vals,color=['#146c94','#75a9bf','#8a99a4'])
for b,v in zip(bars,vals):axes[1].text(b.get_x()+b.get_width()/2,v*1.08,f'{v:.2f}s',ha='center',fontsize=10)
axes[1].set_yscale('log');axes[1].set_ylim(min(vals)*.5,max(vals)*2);axes[1].set_xticks(range(3),['GPU / GPU','GPU / CPU','CPU / CPU']);axes[1].set_ylabel('Document prefill seconds (log scale)');axes[1].set_title('131,072 document tokens: one measured prefill')
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('RWKV-7 2.9B + trained ROSA: current implementation placement timings')
fig.text(.5,.025,'One H80080GB; CPU uses one48-core Xeon8468 NUMA node. GPU body BF16 /CPU body FP32.\nCPU SAM/LCT in all placements; hybrid dynamic gain uses GPU head. Model loading and JIT excluded.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.13,1,.93));fig.savefig(folder/'comparison.png',dpi=180)
