"""Long identifiers/questions and long-answer paired retrieval cases."""
import hashlib
import random

import torch

from .needle_eval import clean_decode


ALPHABET='ABCDEFGHJKLMNPQRSTUVWXYZ'


def answer_text(tokenizer,seed,kind,nominal):
    rng=random.Random(seed+7331)
    if kind=='code':
        # Sixteen/64/128 answer tokens, with each random digit pair checked
        # rather than assuming the vocabulary always tokenizes by two digits.
        value=''.join(str(rng.randrange(10)) for _ in range(2*nominal))
        if len(tokenizer.encode(value))!=nominal:
            raise ValueError('Digit tokenization differs from declared answer length')
        return value
    if kind!='text':raise ValueError('Unknown answer kind')
    clauses=[]
    choices=[('执行节点','青岚站','白榆站','星湾站','松原站'),
             ('核验路径','东侧通道','西侧通道','北侧通道','南侧通道'),
             ('交接位置','三号库房','七号库房','二号库房','九号库房'),
             ('处理动作','先复核编号再封存','先清点标签再交接','先拍照留档再签收','先登记批次再入库')]
    while True:
        name,*options=rng.choice(choices)
        clauses.append(name+'为'+rng.choice(options))
        # Unique random checksum makes an exact answer distinguishable from
        # similar distractors even when some natural-language clauses coincide.
        value='；'.join(clauses)+'；复核码'+str(rng.randrange(10000000,100000000))
        if len(tokenizer.encode(value))>=nominal:return value


def build_long_case(tokenizer,length,depth,seed,kind,nominal):
    rng=random.Random(seed)
    key=''.join(rng.choice(ALPHABET) for _ in range(32))
    batch=str(rng.randrange(100,1000))
    target=answer_text(tokenizer,seed,kind,nominal)
    def anchor(identifier):
        return f'\n档案：天衡归档计划／东部节点／第{batch}批／编号[{identifier}]\n核验内容：\n'
    prefix=anchor(key)
    prefix_ids=tokenizer.encode(prefix);answer_ids=tokenizer.encode(target)
    needle=tokenizer.encode(prefix+target+'。\n结束档案。\n')
    if needle[:len(prefix_ids)]!=prefix_ids or needle[len(prefix_ids):len(prefix_ids)+len(answer_ids)]!=answer_ids:
        raise ValueError('Target token boundary differs in natural document')
    header=tokenizer.encode('User: 请阅读下面的归档记录。记录格式相似，但每份档案的核验内容不同。\n文档开始：\n')
    budget=length-len(header)-len(needle)
    if budget<256:raise ValueError('Insufficient document budget')
    filler=[];boundaries=[0];used={key};number=0
    while True:
        number+=1
        other=''.join(rng.choice(ALPHABET) for _ in range(32))
        while other in used:other=''.join(rng.choice(ALPHABET) for _ in range(32))
        used.add(other)
        payload=answer_text(tokenizer,seed+number*100003,kind,nominal)
        record=tokenizer.encode(anchor(other)+payload+'。\n结束档案。\n')
        if len(filler)+len(record)>budget:break
        filler.extend(record);boundaries.append(len(filler))
    # Insert only at a complete archive boundary. ASCII spaces keep the exact
    # requested token budget without cutting a multibyte character in half.
    at=min(boundaries,key=lambda x:abs(x-budget*depth))
    space=tokenizer.encode(' ')[0]
    filler.extend([space]*(budget-len(filler)))
    stem=header+filler[:at]+needle+filler[at:]
    assert len(stem)==length
    decoded=tokenizer.decode(stem)
    if decoded.count(target)!=1 or decoded.count(key)!=1:
        raise ValueError('Target answer or identifier is not unique')
    qa=tokenizer.encode(f'\n文档结束。\n请从以上大量同类档案中定位唯一符合全部条件的记录：项目为天衡归档计划，区域为东部节点，批次为第{batch}批，完整编号为[{key}]。请逐字返回这份档案的“核验内容”字段全文，保留原有字母、数字、顺序和标点，不要只给开头，不要省略，也不要添加解释。\n\nAssistant:')
    completion=tokenizer.encode('\n文档结束。请逐字补全下面指定档案的完整核验内容，不要省略或添加解释。\n\nAssistant:')+prefix_ids
    for suffix in (qa,completion):
        if target in tokenizer.decode(suffix):raise ValueError('Answer leaked into query')
    return {'id':f'long_{kind}_A{nominal}_L{length}_D{depth:g}_S{seed}',
        'length':length,'depth':depth,'seed':seed,'difficulty':'similar_long_archives',
        'prompt_template':'rwkv','key':key,'secret':target,'answer_ids':answer_ids,
        'answer_kind':kind,'nominal_answer_tokens':nominal,'answer_chars':len(target),
        'actual_answer_tokens':len(answer_ids),'stem_ids':stem,'suffix_ids':{'qa':qa,'exact_prefix':completion},
        'needle_start':len(header)+at,'needle_tokens':len(needle),'actual_document_depth':(len(header)+at)/length,
        'complete_distractor_records':number-1,
        'stem_sha256':hashlib.sha256(torch.tensor(stem,dtype=torch.int64).numpy().tobytes()).hexdigest()}


def long_answer_scores(tokenizer,text,target):
    unclosed='<think>' in text and '</think>' not in text
    final='' if unclosed else text.rsplit('</think>',1)[-1].strip()
    target_ids=tokenizer.encode(target);output_ids=tokenizer.encode(final) if final else []
    best=0
    for start,token in enumerate(output_ids):
        if token!=target_ids[0]:continue
        count=0
        while count<len(target_ids) and start+count<len(output_ids) and output_ids[start+count]==target_ids[count]:count+=1
        best=max(best,count)
    return {'unclosed_thinking':unclosed,'final_answer_text':final,
        'whole_final_exact_match':final==target,
        'final_contains_full_answer':target in final,
        'content_contains_full_answer':target in text,
        'best_correct_prefix_tokens':best,'best_correct_prefix_fraction':best/len(target_ids)}
