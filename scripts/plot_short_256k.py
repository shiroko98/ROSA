from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder=Path(sys.argv[1]);r=json.loads((folder/'summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,mode,cap,title in [(axes[0],'qa',512,'Natural QA (512-token cap)'),(axes[1],'exact_prefix',128,'Exact-prefix completion (128-token cap)')]:
    groups=sorted([g for g in r['comparison'] if g['mode']==mode and g['output_budget']==cap],key=lambda g:g['length'])
    for offset,arm,color,label in [(-.18,'base','#8b99a4','Base'),(.18,'trained','#146c94','ROSA step1800')]:
        values=[100*g[arm]['accuracy'] for g in groups];bars=ax.bar(np.arange(len(groups))+offset,values,.36,label=label,color=color)
        for bar,g in zip(bars,groups):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,f"{g[arm]['final_hit']}/15",ha='center',fontsize=9)
    ax.set_xticks(range(len(groups)),[str(g['length']//1024)+'K'+('*' if g['length']==262144 else '') for g in groups]);ax.set_ylim(0,108)
    ax.set_title(title);ax.set_xlabel('Document tokens');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('Final-answer retrieval accuracy (%)');axes[0].legend(frameon=False)
fig.suptitle('Original short questions: 6-digit answers (3 tokens), same 15 seed/position families')
fig.text(.5,.025,'Greedy; ordinary RWKV User/Assistant format; first standalone passkey in final text.\n*256K extrapolates beyond192,512 training tokens; 128-token /4096-token QA sensitivity is exported separately.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.13,1,.93));fig.savefig(folder/'comparison.png',dpi=180)
