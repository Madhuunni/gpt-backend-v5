"""Shared tokenizer, labels and exact-value decoder (numeric IDs only)."""
import re
LABELS = ['O', 'case_id', 'employee_id', 'sr_no']
FIELDS = LABELS[1:]
# Keep an alphanumeric/decimal/hyphenated value together: never extract a
# numeric substring from an unsupported identifier or date.
TOKEN = re.compile(r'[A-Za-z0-9_]+(?:[./-][A-Za-z0-9_]+)*|[^\w\s]', re.UNICODE)

def tokenize(text):
    return list(TOKEN.finditer(text))

def normalized(token):
    return '<NUM>' if re.fullmatch(r'[0-9]+', token) else token.lower()

def empty_result():
    return dict.fromkeys(FIELDS, '')

def decode(text, matches, predictions, confidences, threshold):
    result = empty_result()
    for match, label, confidence in zip(matches, predictions, confidences):
        value = match.group()
        if label and confidence >= threshold and re.fullmatch(r'[0-9]+', value):
            field = LABELS[label]
            if not result[field]:  # First accepted mention per field wins.
                result[field] = text[match.start():match.end()]
    return result

def labels_for(row):
    matches = tokenize(row['text'])
    labels = [0] * len(matches)
    occupied = set()
    for entity in row['entities']:
        start, end, label = entity['start'], entity['end'], entity['label']
        if label not in FIELDS or not 0 <= start < end <= len(row['text']):
            raise ValueError(f'Invalid entity: {entity}')
        aligned = [i for i,m in enumerate(matches) if m.start() == start and m.end() == end]
        if len(aligned) != 1 or not re.fullmatch(r'[0-9]+', row['text'][start:end]):
            raise ValueError('Entities must align to a complete numeric token')
        i = aligned[0]
        if i in occupied:
            raise ValueError('Overlapping entities')
        labels[i] = LABELS.index(label)
        occupied.add(i)
    return matches, labels

def expected(row):
    matches, labels = labels_for(row)
    return decode(row['text'], matches, labels, [1.] * len(labels), 0.)
