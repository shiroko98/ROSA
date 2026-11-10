#!/usr/bin/env python3
"""Plot completed retrieval results without confusing partial and full answers."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);r=json.loads((out/'summary.json').read_text())
    assert r['status']=='complete'
    lengths=[8192,32768,131072,262144]
    colors=['#8796a5','#146c94']
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,mode,title in zip(axes,['qa','exact_prefix'],['Natural question answering','Exact-prefix completion (copy diagnostic)']):
        gg={g['length']:g for g in r['groups'] if g['group_fields']==['length','mode'] and g['mode']==mode}
        for i,arm in enumerate(['base','trained']):
            values=[100*gg[n][arm]['final_contains_full_answer_rate'] for n in lengths]
            bars=ax.bar(np.arange(4)+(i-.5)*.35,values,.35,label=['Base','ROSA step1800'][i],color=colors[i])
            for bar,n,value in zip(bars,lengths,values):
                count=gg[n][arm]['final_contains_full_answer_correct']
                ax.text(bar.get_x()+bar.get_width()/2,value+1.5,f'{count}/36',ha='center',fontsize=9)
        ax.set_xticks(range(4),['8K','32K','128K','256K*']);ax.set_ylim(0,108)
        ax.set_title(title,fontsize=12);ax.set_xlabel('Document tokens (query added separately)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Full target span in final answer (%)');axes[0].legend(frameon=False)
    fig.suptitle('Frozen RWKV-7 2.9B: longer questions and 16 / 64 / 128-token answers',fontsize=14)
    fig.text(.5,.025,'Greedy; QA budget 4096, prefix budget 1024. *256K extrapolates beyond 192,512-token training context.\nNo partial-copy credit; unclosed thinking is not a final answer. 36 answer/position families repeated across lengths.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.93));fig.savefig(out/'full_answer_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True)
    for row,mode in enumerate(['qa','exact_prefix']):
        for col,arm in enumerate(['base','trained']):
            ax=axes[row,col];matrix=np.zeros((4,6),dtype=int)
            for g in r['groups']:
                if g['group_fields']!=['length','mode','answer_kind','nominal_answer_tokens'] or g['mode']!=mode:continue
                x=(0 if g['answer_kind']=='code' else 3)+[16,64,128].index(g['nominal_answer_tokens'])
                matrix[lengths.index(g['length']),x]=g[arm]['final_contains_full_answer_correct']
            img=ax.imshow(matrix,vmin=0,vmax=6,cmap='Blues',aspect='auto')
            for (y,x),v in np.ndenumerate(matrix):ax.text(x,y,f'{v}/6',ha='center',va='center',color='white' if v>=4 else '#13202d')
            ax.set_xticks(range(6),['digits\n16','digits\n64','digits\n128','text\n16+','text\n64+','text\n128+'])
            ax.set_yticks(range(4),['8K','32K','128K','256K*']);ax.tick_params(labelbottom=True,labelleft=True)
            ax.set_title(('Natural QA' if row==0 else 'Exact-prefix completion')+' / '+('Base' if arm=='base' else 'ROSA'))
            ax.set_xlabel('Answer type and target tokens');ax.set_ylabel('Document tokens')
    fig.suptitle('Complete-answer hits by target length (6 cases per cell)',fontsize=15)
    fig.text(.5,.025,'Structured Chinese text is synthetic field retrieval. Cells are exploratory small samples.\nThe same target strings and insertion depths are paired across all four context lengths.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(out/'answer_length_matrix.png',dpi=180);plt.close(fig)
    print(out/'full_answer_comparison.png')


if __name__=='__main__':main()
