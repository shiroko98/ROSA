"""Loss-preserving virtual windows over the native SFT loader's physical split.

No repacking/copying of the 114GB corpus, and no silent suffix truncation. Every
original target position is assigned to exactly one window. State resets between
windows, explicitly making shorter-context training different from full records.
"""
from __future__ import annotations
import numpy as np


def configure_windowed_dataset(dataset):
    if not getattr(dataset.args,'rosa_windowed_data',0):return dataset
    if dataset.data_type!='sft_binidx':raise ValueError('Windowing only supports sft_binidx.')
    sizes=dataset.data.sizes
    if len(np.unique(sizes))!=1:raise ValueError('Virtual windowing currently requires uniform physical row lengths.')
    context=dataset.args.ctx_len
    if context<=0:raise ValueError('Positive context required.')
    dataset.rosa_physical_indices=(np.array(dataset.sft_doc_indices,copy=True) if dataset.sft_doc_indices is not None
        else np.arange(dataset.sft_doc_start,dataset.sft_doc_start+dataset.sft_doc_count,dtype=np.int64))
    dataset.rosa_windows_per_row=(int(sizes[0])-1+context-1)//context
    dataset.sft_doc_indices=None;dataset.sft_doc_start=0
    dataset.sft_doc_count=len(dataset.rosa_physical_indices)*dataset.rosa_windows_per_row
    print(f'ROSA VIRTUAL WINDOWS: split={dataset.sft_split} physical_rows={len(dataset.rosa_physical_indices)} '
          f'windows_per_row={dataset.rosa_windows_per_row} total_windows={dataset.sft_doc_count} ctx={context}; '
          'all targets preserved, physical split before windows, state reset per window',flush=True)
    return dataset


def read_window(dataset, logical_index):
    physical_offset,slot=divmod(int(logical_index),dataset.rosa_windows_per_row)
    row=int(dataset.rosa_physical_indices[physical_offset]);start=slot*dataset.args.ctx_len
    length=min(dataset.args.ctx_len+1,int(dataset.data.sizes[row])-start)
    ids=dataset.data.get(row,offset=start,length=length).astype(int)
    masks=dataset.mask_data.get(row,offset=start,length=length).astype(int)
    if len(ids)<2:raise ValueError('Virtual window has no target.')
    return ids,masks
