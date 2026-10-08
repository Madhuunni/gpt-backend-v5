import argparse
import json
from pathlib import Path
import torch
from .common import LABELS, tokenize, normalized, empty_result, decode
from .model import IDTagger

class IDExtractor:
    def __init__(self, model_dir='artifacts', device='cpu', threshold=None):
        path=Path(model_dir)
        metadata=json.loads((path/'metadata.json').read_text())
        if metadata['labels'] != LABELS:
            raise ValueError('Incompatible label mapping')
        self.vocab=metadata['vocab']; self.device=torch.device(device)
        self.threshold=metadata['threshold'] if threshold is None else threshold
        if not 0 <= self.threshold <= 1: raise ValueError('threshold must be between 0 and 1')
        self.model=IDTagger(len(self.vocab), **metadata['model_config']).to(self.device)
        self.model.load_state_dict(torch.load(path/'model.pt',map_location=self.device,weights_only=True))
        self.model.eval()

    @torch.inference_mode()
    def predict(self, query):
        if not isinstance(query,str): raise TypeError('query must be a string')
        matches=tokenize(query)
        if not matches: return empty_result()
        if len(matches)>512: raise ValueError('Query exceeds the 512-token limit; split it first')
        if not any(m.group().isascii() and m.group().isdigit() for m in matches): return empty_result()
        ids=[self.vocab.get(normalized(m.group()),1) for m in matches]
        logits=self.model(torch.tensor([ids],device=self.device),torch.tensor([len(ids)]))
        confidence, labels=logits.softmax(-1)[0].max(-1)
        return decode(query,matches,labels.tolist(),confidence.tolist(),self.threshold)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model-dir',default='artifacts'); p.add_argument('--query',required=True)
    p.add_argument('--threshold',type=float); a=p.parse_args()
    print(json.dumps(IDExtractor(a.model_dir,threshold=a.threshold).predict(a.query)))
