"""Bootstrap span labels from the supplied patterns; review before production use."""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path
from .common import tokenize, normalized, labels_for

CUES = {
    'case_id': r'\bcase\s*(?:(?:id|number|no\.?)\s*)?',
    'employee_id': r'\b(?:employee|emp)\s*(?:(?:id|number|no\.?)\s*)?',
    'sr_no': r'\b(?:op\s+)?(?:sr|service\s+request)\s*(?:(?:id|number|no\.?)\s*)?',
}
RULES = {k: re.compile(v + r'[:#=\-]?\s*([0-9]+)\b(?![./-][A-Za-z0-9])', re.I) for k,v in CUES.items()}

def annotate(text):
    entities = sorted(({'start':m.start(1), 'end':m.end(1), 'label':label}
                       for label, rule in RULES.items() for m in rule.finditer(text)),
                      key=lambda e: e['start'])
    row = {'text': text, 'entities': entities}
    labels_for(row)
    return row

def family(text):
    # Group identical wording across ALL ID values AND entity cue variants.
    text = text.lower()
    for rule in RULES.values():
        text = rule.sub('entity <value>', text)
    return ' '.join(normalized(m.group()) for m in tokenize(text))

def split_for(key):
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8],16) % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'

def prepare(source, output):
    data = json.loads(Path(source).read_text())
    raw = list(dict.fromkeys(s for item in data['intents'] for s in item['query_patterns']))
    rows = []
    for text in raw:
        row = annotate(text)
        labeled = {(e['start'],e['end']) for e in row['entities']}
        numbers = {(m.start(),m.end()) for m in tokenize(text) if m.group().isdigit()}
        if numbers - labeled:
            raise ValueError(f'Unlabeled number in source; annotate manually: {text}')
        rows.append((family(text), row))
    # Additional cue diversity and multi-entity examples, generated before splitting.
    frames = ['Show {}', 'Please retrieve {}', 'Find the latest record for {}',
              'Can you check {}?', 'I need information about {}', '{}: get the details',
              'Look up {} for me', 'Return the report associated with {}',
              'What is the status of {}?', 'Pull up {}', 'Please process {}',
              'Give me an update on {}', 'Fetch information on {}', 'Display {}',
              'Could you locate {}?', 'I want to view {}']
    cues = ['case ID', 'case number', 'case no.', 'case', 'employee ID',
            'employee number', 'employee', 'emp ID', 'SR number', 'SR no.',
            'SR', 'OP SR', 'service request number']
    for frame in frames:
        for cue in cues:
            for separator in [' ', ': ', ' #']:
                text = frame.format(cue + separator + '0012345')
                rows.append(('extra-positive:' + frame, annotate(text)))
        for clause in ['case ID 00123 and employee ID 00456 and SR number 00789',
                       'employee 00456 linked to case 00123', 'SR 00789 for employee 00456']:
            text = frame.format(clause)
            rows.append(('extra-positive:' + frame, annotate(text)))
    negatives = ['Show the latest letter', 'There are 12 records', 'Show the last 5 cases',
                 'List 20 employees', 'Get reports from 2024', 'Amount is 5000 rupees',
                 'Show 3 letters for the employee', 'Case ID is missing', 'Employee ID',
                 'SR number not available', 'Hello', '12345', 'Get details for 12345',
                 'Show overpayment details', 'Report dated 2025-03-29',
                 'The case amount is 5000', 'The employee age is 42', 'SR count is 8',
                 'There are 100 open cases', 'Show employee salary 45000',
                 'Get the last 10 SR records', 'Employee joined in 2020',
                 'Case pending for 30 days', 'Show page 2', 'Download 7 reports']
    for text in negatives:
        for prefix in ['', 'Please: ', 'Can you help? ']:
            rows.append(('negative:' + text, {'text':prefix+text, 'entities':[]}))
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    splits = collections.defaultdict(list); seen = set(); rng = random.Random(42)
    for key, row in rows:
        if row['text'] in seen: continue
        seen.add(row['text']); split = split_for(key)
        row['group'] = key
        splits[split].append(row)
        # Randomized values only AFTER group assignment; no cross-split copies.
        if row['entities']:
            for _ in range(2):
                text = row['text']
                for e in reversed(row['entities']):
                    value = ''.join(rng.choice('0123456789') for _ in range(rng.randint(3,12)))
                    text = text[:e['start']] + value + text[e['end']:]
                variant = annotate(text); variant['group'] = key
                splits[split].append(variant)
    for name, items in splits.items():
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in items))
    stats = {'source_patterns':len(raw), 'splits': {k:len(v) for k,v in splits.items()},
             'entity_counts':{k:dict(collections.Counter(e['label'] for r in v for e in r['entities'])) for k,v in splits.items()},
             'note':'Rule-bootstrapped and synthetic labels; template-family split, not independently human-labeled evaluation.'}
    (out/'data_summary.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats,indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--input', default='data/query_patterns.json'); p.add_argument('--output',default='data')
    a=p.parse_args(); prepare(a.input,a.output)
