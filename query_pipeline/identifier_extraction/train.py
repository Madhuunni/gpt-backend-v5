import argparse
import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .common import LABELS, normalized, labels_for, expected, decode, FIELDS
from .model import IDTagger

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def make_batch(rows,vocab):
    encoded=[]; targets=[]
    for row in rows:
        matches, labels=labels_for(row)
        if not matches or len(matches)>512: raise ValueError('Training text must have 1 to 512 tokens')
        encoded.append([vocab.get(normalized(m.group()),1) for m in matches]); targets.append(labels)
    lengths=torch.tensor([len(x) for x in encoded]); width=int(lengths.max())
    x=torch.zeros(len(rows),width,dtype=torch.long); y=torch.full_like(x,-100)
    for i,(ids,labels) in enumerate(zip(encoded,targets)):
        x[i,:len(ids)]=torch.tensor(ids); y[i,:len(ids)]=torch.tensor(labels)
    return x,y,lengths

@torch.inference_mode()
def collect(model,rows,vocab,device):
    model.eval(); result=[]
    for start in range(0,len(rows),128):
        batch=rows[start:start+128]; x,_,lengths=make_batch(batch,vocab)
        confidence,labels=model(x.to(device),lengths).softmax(-1).max(-1)
        for i,row in enumerate(batch):
            n=int(lengths[i]); result.append((row,labels[i,:n].tolist(),confidence[i,:n].tolist()))
    return result

def metrics(predictions,threshold):
    exact=0; tp=fp=fn=0; negative_total=negative_correct=0
    fields={f:{'tp':0,'fp':0,'fn':0} for f in FIELDS}
    for row,labels,confidence in predictions:
        matches,_=labels_for(row); truth=expected(row)
        pred=decode(row['text'],matches,labels,confidence,threshold)
        exact+=pred==truth
        if not any(truth.values()):
            negative_total+=1; negative_correct+=not any(pred.values())
        for field in FIELDS:
            correct=bool(pred[field]) and pred[field]==truth[field]
            fields[field]['tp']+=int(correct)
            fields[field]['fp']+=int(bool(pred[field]) and not correct)
            fields[field]['fn']+=int(bool(truth[field]) and not correct)
    for counts in fields.values():
        a,b,c=counts['tp'],counts['fp'],counts['fn']; tp+=a; fp+=b; fn+=c
        counts.update(precision=a/max(1,a+b),recall=a/max(1,a+c),f1=2*a/max(1,2*a+b+c))
    return {'examples':len(predictions),'exact_json_accuracy':exact/max(1,len(predictions)),
            'entity_precision':tp/max(1,tp+fp),'entity_recall':tp/max(1,tp+fn),'entity_f1':2*tp/max(1,2*tp+fp+fn),
            'no_id_examples':negative_total,'no_id_accuracy':negative_correct/negative_total if negative_total else None,
            'per_field':fields}

def main(a):
    random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    if a.epochs<1: raise ValueError('epochs must be positive')
    train=read_rows(Path(a.data)/'train.jsonl'); valid=read_rows(Path(a.data)/'validation.jsonl')
    if not train or not valid: raise ValueError('Nonempty training and validation sets are required')
    if {r['group'] for r in train} & {r['group'] for r in valid}: raise ValueError('Group leakage')
    vocab={'<PAD>':0,'<UNK>':1,'<NUM>':2}
    for row in train:
        for match in labels_for(row)[0]:
            word=normalized(match.group())
            if word not in vocab: vocab[word]=len(vocab)
    config={'embedding_dim':48,'hidden_dim':64}; device=torch.device(a.device)
    model=IDTagger(len(vocab),**config).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=0.003)
    loss_fn=torch.nn.CrossEntropyLoss(ignore_index=-100,weight=torch.tensor([1.,5.,5.,5.],device=device))
    loader=DataLoader(train,batch_size=64,shuffle=True,collate_fn=lambda rows:make_batch(rows,vocab))
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); best=-1.; history=[]
    for epoch in range(1,a.epochs+1):
        model.train(); loss_total=0.
        for x,y,lengths in loader:
            optimizer.zero_grad(); logits=model(x.to(device),lengths)
            loss=loss_fn(logits.reshape(-1,4),y.to(device).reshape(-1)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); loss_total+=loss.item()
        score=metrics(collect(model,valid,vocab,device),0.5)['exact_json_accuracy']
        log={'epoch':epoch,'loss':loss_total/len(loader),'validation_exact_json_accuracy':score}; history.append(log); print(json.dumps(log),flush=True)
        if score>best:
            best=score; torch.save(model.state_dict(),out/'model.pt')
    model.load_state_dict(torch.load(out/'model.pt',map_location=device,weights_only=True))
    predictions=collect(model,valid,vocab,device)
    candidates=[(t,metrics(predictions,t)) for t in [0.5,0.6,0.7,0.8,0.9,0.95]]
    threshold,report=max(candidates,key=lambda x:(x[1]['exact_json_accuracy'],x[1]['entity_f1'],x[0]))
    metadata={'vocab':vocab,'labels':LABELS,'model_config':config,'threshold':threshold,
              'tokenizer_version':1,'seed':a.seed,'torch_version':str(torch.__version__),
              'numeric_ids_only':True,'duplicate_policy':'first accepted mention per field'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'training_report.json').write_text(json.dumps({'history':history,'selected_threshold':threshold,'validation':report},indent=2))
    print('Saved model and metadata to',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data'); p.add_argument('--output',default='artifacts')
    p.add_argument('--epochs',type=int,default=12); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=2); p.add_argument('--device',default='cpu'); main(p.parse_args())
