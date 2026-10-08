import argparse
import json
from pathlib import Path
import torch
from .extractor import IDExtractor
from .train import read_rows, collect, metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model-dir',default='artifacts'); p.add_argument('--test',default='data/test.jsonl')
    p.add_argument('--output',default='artifacts/test_report.json'); a=p.parse_args(); torch.set_num_threads(2)
    extractor=IDExtractor(a.model_dir); rows=read_rows(a.test)
    report=metrics(collect(extractor.model,rows,extractor.vocab,extractor.device),extractor.threshold)
    report['note']='Held-out template-family split of bootstrapped/synthetic examples; not a production accuracy guarantee.'
    Path(a.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
